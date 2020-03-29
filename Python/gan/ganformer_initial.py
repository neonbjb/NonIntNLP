import os
import random
import torch
from torch import nn
import torch.utils.data
import transformers
import wandb
import tqdm
from dataloaders.chunked_text_dataloader import ChunkedTextDataset
from apex import amp

def main():
    # Set random seed for reproducibility
    random.seed(999)
    torch.manual_seed(999)

    # Root directory for dataset
    dataroot = "C:\\Users\\jbetk\\Documents\\data\\ml\\xsum\\xsum-extracts-from-downloads\\outputs"
    output_dir = "C:\\Users\\jbetk\\Documents\\data\\ml\\saved_models\\ganformer"
    # Batch size during training
    batch_size = 2
    # Number of training epochs
    num_epochs = 1
    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1
    # Whether or not to log to w&b
    do_wandb = True

    # new stuff
    model_name = "xlnet-base-cased"
    seq_sz = 256
    max_predict_sz = 58
    start_lr = 2e-5
    mem_len = 768


    tokenizer = transformers.XLNetTokenizer.from_pretrained(
        model_name
    )
    dataset = ChunkedTextDataset(
        os.path.join(dataroot, "train.pt"),
        tokenizer,
        seq_sz,
        max_predict_sz,
        pad_left=True,
    )
    dataloader = dataset.get_dataloader(batch_size, random=True, num_workers=0)
    configG = transformers.XLNetConfig.from_pretrained(
        model_name
    )
    configG.mem_len = mem_len
    configG.output_hidden_states = 1
    configD = transformers.XLNetConfig.from_pretrained(
        model_name
    )
    configD.mem_len = mem_len
    configD.num_labels = 1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Create the generator
    netG = transformers.XLNetLMHeadModel.from_pretrained(model_name, config=configG).to(device)
    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Create the Discriminator
    netD = transformers.XLNetForSequenceClassification.from_pretrained(model_name, config=configD).to(device)
    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    def save_models(chkpt):
        chkptG_dir = os.path.join(output_dir, "chkpt_%i/generator" % (chkpt))
        if not os.path.exists(chkptG_dir):
            os.makedirs(chkptG_dir)
        sgModel = (
            netG if hasattr(netG, "module") else netG
        )  # Take care of distributed/parallel training
        sgModel.save_pretrained(chkptG_dir)
        chkptD_dir = os.path.join(output_dir, "chkpt_%i/discriminator" % (chkpt))
        if not os.path.exists(chkptD_dir):
            os.makedirs(chkptD_dir)
        sdModel = (
            netD if hasattr(netD, "module") else netD
        )  # Take care of distributed/parallel training
        sdModel.save_pretrained(chkptD_dir)
    # Validate save_models works.
    #save_models(0)

    # Initialize BCELoss function
    criterion = nn.BCEWithLogitsLoss()

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    def get_opt_and_sched(model):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=start_lr, eps=1e-8)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=num_epochs * len(dataset),
        )
        return optimizer, scheduler

    # Setup Adam optimizers for both G and D
    optimizerD, schedulerD = get_opt_and_sched(netD)
    optimizerG, schedulerG = get_opt_and_sched(netG)

    models, optimizers = amp.initialize([netG, netD], [optimizerG, optimizerD], opt_level="O1")
    netG, netD = models
    optimizerG, optimizerD = optimizers

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        its_per_chkpt = 2000
        its_per_log = 5

        it = tqdm.tqdm(dataloader)
        # For each batch in the dataloader
        for batch in it:
            num_chunks = len(batch["input_ids"])

            def computeMemsForModel(model):
                mems = None
                for c in range(num_chunks-1):
                    inputs = {
                        "input_ids": batch["input_ids"][c].to(device),
                        "attention_mask": batch["attention_masks"][c].to(device),
                    }
                    if mems is not None:
                        inputs["mems"] = mems
                    with torch.no_grad():
                        output = model.forward(**inputs)
                        logits, mems = output[0], output[1]
                return mems

            # Compute the mems for both the discriminator and the generator.
            memsD = computeMemsForModel(netD)
            memsG = computeMemsForModel(netG)

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_inputs = {
                "input_ids": batch["input_ids"][-1].to(device),
                "attention_mask": batch["attention_masks"][-1].to(device),
                "mems": memsD
            }
            label = torch.full((batch_size,), real_label, device=device)
            # Forward pass real batch through D
            logits, _ = netD(**real_inputs)
            logits = logits.view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(logits, label)
            # Calculate gradients for D in backward pass
            with amp.scale_loss(errD_real, optimizerD) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                amp.master_params(optimizerD), 1
            )

            D_x = logits.mean().item()

            ## Train with all-fake batch
            # Compute the embeddings and add in noise.
            g_embeddings = netG.get_input_embeddings().forward(batch["input_ids_masked"][-1].to(device))

            # Generate noise to apply across the <masked> tokens.
            noise_magnitude_max = g_embeddings.mean() * .1
            noise_magnitude_min = g_embeddings.mean() * -.1
            target_noise = torch.randn((batch_size, max_predict_sz, mem_len), device=device) * (noise_magnitude_max - noise_magnitude_min) + noise_magnitude_min
            text_noise = torch.zeros((batch_size, seq_sz - max_predict_sz, mem_len), device=device)
            noise = torch.cat([text_noise, target_noise], dim=1)

            g_inputs = {
                "inputs_embeds": g_embeddings + noise,
                "attention_mask": batch["attention_masks"][-1].to(device),
                "mems": memsG
            }
            logits, mems, hidden_states = netG(**g_inputs)
            final_target_embeddings = hidden_states[-1][:, -max_predict_sz:, :]
            initial_text_embeddings = g_embeddings[:, :-max_predict_sz, :]

            discriminator_embeddings = torch.cat([initial_text_embeddings, final_target_embeddings], dim=1)
            fakeDInputs = {
                "inputs_embeds": discriminator_embeddings,
                "attention_mask": batch["attention_masks"][-1].to(device),
                "mems": memsD
            }
            fakeDInputsDPass = {
                "inputs_embeds": discriminator_embeddings.detach(),
                "attention_mask": batch["attention_masks"][-1].to(device).detach(),
                "mems": memsD
            }

            # Generate fake image batch with G
            label.fill_(fake_label)
            # Classify all fake batch with D
            logits, _ = netD(**fakeDInputsDPass)
            logits = logits.view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(logits, label)
            # Calculate the gradients for this batch
            with amp.scale_loss(errD_fake, [optimizerD, optimizerG]) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                amp.master_params(optimizerD), 1
            )

            D_G_z1 = logits.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()
            schedulerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            logits, _ = netD(**fakeDInputs)
            logits = logits.view(-1)
            # Calculate G's loss based on this output
            errG = criterion(logits, label)
            # Calculate gradients for G
            with amp.scale_loss(errG, optimizerG) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                amp.master_params(optimizerG), 1
            )
            D_G_z2 = logits.mean().item()
            # Update G
            optimizerG.step()
            schedulerG.step()

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Output training stats
            if iters != 0 and iters % its_per_log == 0:
                mGL = sum(G_losses[-its_per_log:]) / its_per_log
                mDL = sum(D_losses[-its_per_log:]) / its_per_log
                if do_wandb:
                    log = {
                        'iteration': iters,
                        'epoch': epoch,
                        'discriminatorLoss': mDL,
                        'generatorLoss': mGL,
                        'D_x': D_x,
                        'DG_z1': D_G_z1,
                        'DG_z2': D_G_z2,
                        'lr_D': schedulerD.get_lr()[0],
                        'lr_G': schedulerG.get_lr()[0]
                    }
                    wandb.log(log)
                else:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, num_epochs, iters, len(dataloader),
                             mDL, mGL, D_x, D_G_z1, D_G_z2))


            if iters % its_per_chkpt == 0:
                print("Saving checkpoint at %i" % iters)
                save_models(iters)

            iters += 1

if __name__ == "__main__":
    wandb.init(project="nonint-ganformer-torch",\
               name="ganformer_v1",\
               config={"dataset": "celeba"})
    main()