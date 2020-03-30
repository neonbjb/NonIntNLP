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
    # Whether or not to add noise on the masks for the input embeddings fed into the generator.
    add_noise = False

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
    configD.num_labels = 2
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
    save_models(0)

    # Initialize BCELoss function
    criterion = nn.BCEWithLogitsLoss()

    # Establish convention for real and fake labels during training
    real_label = torch.ones((batch_size,), dtype=torch.long)
    fake_label = torch.zeros((batch_size,), dtype=torch.long)

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
    G_losses = []
    G_lm_losses = []
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

            def forward_backward_both_models(batch, chunk, input_label_to_use, memsG, memsD, gen_labels=None,
                                             disc_labels=None, compute_discriminator_loss=False,
                                             discriminator_to_generator_loss=False, do_noise=False):
                _embeddings = netG.get_input_embeddings().forward(batch[input_label_to_use][chunk].to(device))
                attention_mask = batch["attention_masks"][chunk].to(device)
                _noise = None
                if do_noise:
                    # Generate noise to apply across the <masked> tokens.
                    noise_magnitude_max = _embeddings.mean() * .1
                    noise_magnitude_min = _embeddings.mean() * -.1
                    target_noise = torch.randn((batch_size, max_predict_sz, mem_len), device=device) * (
                                noise_magnitude_max - noise_magnitude_min) + noise_magnitude_min
                    text_noise = torch.zeros((batch_size, seq_sz - max_predict_sz, mem_len), device=device)
                    _noise = torch.cat([text_noise, target_noise], dim=1, device=device)

                g_inputs = {
                    "inputs_embeds": _embeddings + _noise if _noise is not None else _embeddings,
                    "attention_mask": attention_mask,
                    "mems": memsG
                }
                if gen_labels is not None:
                    g_inputs["labels"] = gen_labels.to(device)
                    lossG, logits, memsG, hidden_states = netG(**g_inputs)

                    # Since a loss was provided, compute backwards as well before going on to the discriminator. This
                    # forces the gradients to halt at the output of the generator (desired).
                    with amp.scale_loss(lossG, optimizerG) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizerG), 1
                    )

                    # Detach the hidden_state from the generator graph. We don't want the discriminator to backprop into
                    # the generator.
                    final_hidden_state = hidden_states[-1].detach()
                else:
                    if discriminator_to_generator_loss:
                        logits, memsG, hidden_states = netG(**g_inputs)
                        final_hidden_state = hidden_states[-1]
                    else:
                        # Do not allow gradients to flow into the generator from the discriminator.
                        with torch.no_grad():
                            logits, memsG, hidden_states = netG(**g_inputs)
                            final_hidden_state = hidden_states[-1]

                d_inputs = {
                    "inputs_embeds": final_hidden_state,
                    "attention_mask": attention_mask,
                    "mems": memsD,
                }
                if disc_labels is not None:
                    d_inputs["labels"] = disc_labels.to(device)
                    lossD, logits, memsD = netD(**d_inputs)

                    if compute_discriminator_loss:
                        with amp.scale_loss(lossD, optimizerD) as scaled_loss:
                            scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(optimizerD), 1
                        )
                else:
                    with torch.no_grad():
                        logits, memsD = netD(**d_inputs)

                if discriminator_to_generator_loss:
                    with amp.scale_loss(lossD, optimizerG) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizerG), 1
                    )

                if gen_labels is not None and disc_labels is not None:
                    return lossG, lossD
                elif disc_labels is not None:
                    return lossD
                else:
                    return memsG, memsD

            ############################
            # (0) Compute the mems basis for both graphs.
            ###########################
            memsG, memsD = None, None
            for c in range(num_chunks-1):
                memsG, memsD = forward_backward_both_models(batch, c, "input_ids", memsG, memsD)

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch. In this case we want to induce the generator to "mirror" in the input_ids in
            ## output logits. It should only mutate <mask> tokens.
            lossG, lossD = forward_backward_both_models(batch, -1, "input_ids", memsG, memsD,
                                                        gen_labels=batch["input_ids"][-1],
                                                        disc_labels=real_label,
                                                        compute_discriminator_loss=True,
                                                        do_noise=False)
            G_lm_losses.append(lossG.item())
            D_losses.append(lossD.item())

            ## Train with all-fake batch
            # Compute the embeddings and add in noise.
            lossD = forward_backward_both_models(batch, -1, "input_ids_masked", memsG, memsD,
                                                 gen_labels=None, disc_labels=fake_label,
                                                 compute_discriminator_loss=True,
                                                 do_noise=False)
            D_losses.append(lossD.item())

            # Update D. Note that G still has gradients stored in it - we'll consume those in the next step.
            optimizerD.step()
            schedulerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            lossG = forward_backward_both_models(batch, -1, "input_ids_masked", memsG, memsD, gen_labels=None,
                                                 disc_labels=real_label,
                                                 compute_discriminator_loss=False,
                                                 discriminator_to_generator_loss=True,
                                                 do_noise=False)
            G_losses.append(lossG)

            # Update G
            optimizerG.step()
            schedulerG.step()

            # Clear out the grads. There shouldn't be any in G, but D will have some. #Just be safe.
            netD.zero_grad()
            netG.zero_grad()

            # Output training stats
            if iters != 0 and iters % its_per_log == 0:
                mGL = sum(G_losses[-its_per_log:]) / its_per_log
                mGL_lm = sum(G_lm_losses[-its_per_log:]) / its_per_log
                mDL = sum(D_losses[-its_per_log:]) / its_per_log
                if do_wandb:
                    log = {
                        'iteration': iters,
                        'epoch': epoch,
                        'discriminatorLoss': mDL,
                        'generatorLoss': mGL,
                        'generatorLangModelLoss': mGL_lm,
                        'lr_D': schedulerD.get_lr()[0],
                        'lr_G': schedulerG.get_lr()[0]
                    }
                    wandb.log(log)

            if iters % its_per_chkpt == 0:
                print("Saving checkpoint at %i" % iters)
                save_models(iters)

            iters += 1

if __name__ == "__main__":
    wandb.init(project="nonint-ganformer-torch",\
               name="ganformer_v1",\
               config={"dataset": "celeba"})
    main()