# This source file trains a standalone discriminator to test out whether it is possible to discriminate a dataset
# outside of the GAN topology.

import torch
import transformers
import random
import tqdm
import wandb
from apex import amp

class GeneratedDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.raw_data = torch.load("C:\\Users\\jbetk\\Documents\\data\\ml\\saved_models\\ganformer\\test_chkpt\\generated_data.pt")

    def __getitem__(self, index):
        usefake = bool(random.getrandbits(1))
        if usefake:
            return {"input_ids": torch.cat([torch.tensor(tokenizer.encode(""), dtype=torch.long), self.raw_data[index]["generated_ids"][:-2]]).cuda(), "labels": torch.tensor(1, dtype=torch.long, device="cuda")}
        else:
            return {"input_ids": self.raw_data[index]["input_ids"].cuda(), "labels": torch.tensor(0, dtype=torch.long, device="cuda")}

    def __len__(self):
        return len(self.raw_data)


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
    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=1e-5, eps=1e-8)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=5000
    )
    return optimizer, scheduler

model_name = "albert-large-v2"
tokenizer = transformers.AlbertTokenizer.from_pretrained(
    model_name
)
configD = transformers.AlbertConfig.from_pretrained(
    model_name
)
configD.num_labels = 2
device = torch.device("cuda")
netD = transformers.AlbertForSequenceClassification.from_pretrained(model_name, config=configD).to(device)
optimizer, scheduler = get_opt_and_sched(netD)
netD, optimizer = amp.initialize(netD, optimizer, opt_level="O0")

dataset = GeneratedDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=0, shuffle=True)

wandb.init(project="nonint-ganformer-torch", \
           name="albert-discriminator-only", \
           config={"dataset": "wiki"})

it = tqdm.tqdm(dataloader)
# For each batch in the dataloader
loss_accum = []
steps = 0
for batch in it:
    loss, logits = netD(**batch)
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
    optimizer.step()
    scheduler.step()
    loss_accum.append(loss.item())
    steps += 1
    if steps % 10 == 0:
        wandb.log({
            "loss": sum(loss_accum[-10:]) / 10
        })