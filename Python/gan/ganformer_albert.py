import os
import random
import torch
from torch import nn
import torch.utils.data
import transformers
import wandb
import tqdm
from dataloaders.wiki_mask_dataset import WikiMaskDataset
from apex import amp

def main():
    # Root directory for dataset
    dataroot = "C:\\Users\\jbetk\\Documents\\data\\ml\\wiki\\processed"
    # Output directory for model checkpoints.
    output_dir = "C:\\Users\\jbetk\\Documents\\data\\ml\\saved_models\\ganformer\\albert-wiki"
    # Batch size during training
    batch_size = 4
    # Number of training epochs
    num_epochs = 1
    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1
    # Whether or not to log to w&b
    do_wandb = True

    # new stuff
    model_name = "albert-large-v2"
    disc_preload = model_name # "C:\\Users\\jbetk\\Documents\\data\\ml\\saved_models\\ganformer\\chkpt\\discriminator"
    gen_preload = "C:\\Users\\jbetk\\Documents\\data\\ml\\saved_models\\ganformer\\chkpt\\generator"
    seq_sz = 256
    start_lr = 2e-5

    # For training
    num_masked_tokens = 5
    num_graded_tokens = 10

    tokenizer = transformers.AlbertTokenizer.from_pretrained(
        model_name
    )
    dataset = WikiMaskDataset(
        os.path.join(dataroot, "train.pt"),
        tokenizer,
        seq_sz,
        num_elements_masked=num_masked_tokens
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    configG = transformers.AlbertConfig.from_pretrained(
        model_name
    )
    configG.output_hidden_states = 1
    configD = transformers.AlbertConfig.from_pretrained(
        model_name
    )
    configD.num_labels = 2
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Create the generator
    netG = transformers.AlbertForMaskedLM.from_pretrained(gen_preload, config=configG).to(device)
    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Create the Discriminator
    netD = transformers.AlbertForSequenceClassification.from_pretrained(disc_preload, config=configD).to(device)
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

    if do_wandb:
        wandb.init(project="nonint-ganformer-torch", \
                   name="albert-ganformer", \
                   config={"dataset": "wiki"})

    # Training Loop

    # Lists to keep track of progress
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        its_per_chkpt = 500
        its_per_log = 5

        it = tqdm.tqdm(dataloader)
        # For each batch in the dataloader
        for batch in it:

            # Albert has a special layer between the hidden states and the embeddings. Consider transferring these
            # states across models rather than the embedding states.
            def generator_hidden_states_to_embedding(hidden_states):
                hidden_states = netG.predictions.dense(hidden_states)
                hidden_states = netG.predictions.activation(hidden_states)
                hidden_states = netG.predictions.LayerNorm(hidden_states)
                return hidden_states

            # Possible phases:
            # "disc_real" - Compute discriminator loss against a real sample
            # "disc_fake" - Compute discriminator loss against the hidden state from the generator given a masked input
            # "gen_fake" - Compute generator loss through the discriminator
            def forward_backward_both_models(batch, phase, disc_labels=None):
                _embeddings = netG.get_input_embeddings().forward(batch["input_ids"].to(device))
                _embeddings_masked = netG.get_input_embeddings().forward(batch["input_ids_masked"].to(device))

                # We don't need the generator in the "disc_real" phase.
                if phase != "disc_real":
                    g_inputs = {
                        "inputs_embeds": _embeddings_masked
                    }
                    # Only compute generator gradients when we are going to backprop through the generator.
                    if phase == "gen_fake":
                        logitsG, hidden_states = netG(**g_inputs)
                        final_hidden_state = generator_hidden_states_to_embedding(hidden_states[-1])
                        computedLogits = netG.get_output_embeddings().forward(final_hidden_state)
                    else:
                        # Do not allow gradients to flow into the generator from the discriminator when training the discriminator.
                        with torch.no_grad():
                            logitsG, hidden_states = netG(**g_inputs)
                            final_hidden_state = generator_hidden_states_to_embedding(hidden_states[-1])
                            computedLogits = netG.get_output_embeddings().forward(final_hidden_state)

                if phase == "disc_fake" or phase == "gen_fake":
                    disc_embeddings = torch.cat([_embeddings[:, :-num_graded_tokens, :], final_hidden_state[:, -num_graded_tokens:, :]], dim=1)

                    #diff = disc_embeddings - _embeddings
                    #_embeddingsLogits = netG.get_output_embeddings().forward(_embeddings)
                    #print("\nEmb: " + tokenizer.decode(_embeddingsLogits.softmax(-1).argmax(-1)[0]))
                    #print("Gen: " + tokenizer.decode(computedLogits.softmax(-1).argmax(-1)[0]))
                    # Average differences across each sequence.
                    #diff = diff.sum(dim=-1) / diff.shape[-2]
                    #print("Embedding differences:")
                    #for jb in range(batch_size):
                    #    print("[%i]: [%s]" % (jb, str(diff[jb].cpu())))
                else:
                    disc_embeddings = _embeddings

                d_inputs = {
                    "inputs_embeds": disc_embeddings,
                    "labels": disc_labels.to(device)
                }
                lossD, logits = netD(**d_inputs)

                if phase == "disc_real" or phase == "disc_fake":
                    with amp.scale_loss(lossD, optimizerD) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizerD), 1
                    )

                if phase == "gen_fake":
                    with amp.scale_loss(lossD, optimizerG) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizerG), 1
                    )

                return lossD

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch. In this case we want to induce the generator to "mirror" in the input_ids in
            ## output logits. It should only mutate <mask> tokens.
            lossD = forward_backward_both_models(batch, disc_labels=real_label, phase="disc_real")
            D_losses.append(lossD.item())

            ## Train with all-fake batch
            # Compute the embeddings and add in noise.
            lossD = forward_backward_both_models(batch, disc_labels=fake_label, phase="disc_fake")
            D_losses.append(lossD.item())

            # Update D.
            optimizerD.step()
            schedulerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            lossG = forward_backward_both_models(batch, disc_labels=real_label, phase="gen_fake")
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
                mDL = sum(D_losses[-its_per_log:]) / its_per_log
                if do_wandb:
                    log = {
                        'iteration': iters,
                        'epoch': epoch,
                        'discriminatorLoss': mDL,
                        'generatorLoss': mGL,
                        'lr_D': schedulerD.get_lr()[0],
                        'lr_G': schedulerG.get_lr()[0]
                    }
                    wandb.log(log)

            if iters % its_per_chkpt == 0:
                print("Saving checkpoint at %i" % iters)
                save_models(iters)

            iters += 1

if __name__ == "__main__":
    main()