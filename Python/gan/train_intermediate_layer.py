import torch
import torch.nn as nn
import transformers
from dataloaders.chunked_text_dataloader import ChunkedTextDataset
import os
import tqdm
import wandb

# Root directory for dataset
dataroot = "C:\\Users\\jbetk\\Documents\\data\\ml\\xsum\\xsum-extracts-from-downloads\\outputs"
output_dir = "C:\\Users\\jbetk\\Documents\\data\\ml\\saved_models\\ganformer\\converters"
# Batch size during training
batch_size = 32
# Number of training epochs
num_epochs = 100
train = True
train_on_real_data = False

# new stuff
model_name = "xlnet-base-cased"
seq_sz = 256
max_predict_sz = 80
start_lr = 1e-5
device = torch.device("cuda")

tokenizer = transformers.XLNetTokenizer.from_pretrained(
    model_name
)
config = transformers.XLNetConfig.from_pretrained(
    model_name
)
config.output_hidden_states = 1

# Create the generator
xlnet = transformers.XLNetLMHeadModel.from_pretrained(model_name, config=config).to(device)

# Create the conversion layer between generator hidden states and discriminator embedding inputs.
## Uncomment to train from scratch
# conversionLayer = nn.Linear(config.d_model, config.d_model, bias=True).to(device)
# xlnet._init_weights(conversionLayer)
conversionLayer = torch.load("C:\\Users\\jbetk\\Documents\\data\\ml\\saved_models\\ganformer\\converters\\encoder_decoder.pt")

encoder = xlnet.get_input_embeddings()
decoder = xlnet.get_output_embeddings()

loss_fct = nn.CrossEntropyLoss()

optimizer_grouped_parameters = [{
    "params": [p for n, p in conversionLayer.named_parameters()],
    "weight_decay": 0,
}]
optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=start_lr, eps=1e-8)
scheduler = transformers.get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=num_epochs * 1000
)

if train:
    wandb.init(project="nonint-ganformer-torch", \
               name="train_embedding_converter", \
               config={"dataset": "xsum"})


class BasicIterableDataset:
    def __init__(self, size):
        self.size = size

    def __getitem__(self, index):
        return torch.randint(0, self.size, (seq_sz,), dtype=torch.long)

    def __len__(self):
        return self.size

token_set = BasicIterableDataset(len(tokenizer.get_vocab()))
token_loader = torch.utils.data.DataLoader(token_set, batch_size=batch_size, num_workers=0, shuffle=True)

dataset = ChunkedTextDataset(
    os.path.join(dataroot, "train.pt"),
    tokenizer,
    seq_sz,
    max_predict_sz,
    add_pads_to_target=False
)
dataloader = dataset.get_dataloader(batch_size, random=True, num_workers=0)

steps = 0
for epoch in range(num_epochs):
    loss_accum = []
    if train and not train_on_real_data:
        it = tqdm.tqdm(token_loader)
        for batch in it:
            batch = batch.to(device)
            encoder.zero_grad()
            decoder.zero_grad()
            encoded = encoder(batch)
            encoded = conversionLayer(encoded)
            decoded = decoder(encoded)

            loss = loss_fct(decoded.view(-1, decoded.size(-1)), batch.view(-1))
            loss_accum.append(loss.item())
            loss.backward()

            optimizer.step()
            scheduler.step()
            steps += 1

            if steps % 50 == 0:
                avg_loss = sum(loss_accum[-40:]) / 50
                accuracy = torch.eq(batch, decoded.softmax(-1).argmax(-1)).int().sum().float() / (batch_size * seq_sz)

                wandb.log({"loss": avg_loss,
                           "accuracy": accuracy,
                           "epoch": epoch,
                           "step": steps,
                           "learning_rate": scheduler.get_lr()})
        torch.save(conversionLayer, os.path.join(output_dir, "chkpt_%i.pt" % (epoch)))

    # In train_on_real_data=False mode, this loop performs validation. Otherwise it does the actual training.
    it = tqdm.tqdm(dataloader)
    for batch in it:
        for chunk in batch["input_ids"]:
            chunk = chunk.to(device)
            encoder.zero_grad()
            decoder.zero_grad()

            encoded = encoder(chunk)
            encoded = conversionLayer(encoded)
            decoded = decoder(encoded)

            loss = loss_fct(decoded.view(-1, decoded.size(-1)), chunk.view(-1))

            if train_on_real_data:
                loss_accum.append(loss.item())
                loss.backward()
                optimizer.step()
                scheduler.step()
                steps += 1

                if steps % 50 == 0:
                    avg_loss = sum(loss_accum[-40:]) / 50
                    accuracy = torch.eq(chunk, decoded.softmax(-1).argmax(-1)).int().sum().float() / (batch_size * seq_sz)

                    wandb.log({"loss": avg_loss,
                               "accuracy": accuracy,
                               "epoch": epoch,
                               "step": steps,
                               "learning_rate": scheduler.get_lr()})

            if not train_on_real_data or steps % 500 == 0:
                for b in range(batch_size):
                    avg_loss = sum(loss_accum[-40:]) / 50
                    print("\nBEFOR: " + tokenizer.decode(chunk[b]))
                    print("AFTER: " + tokenizer.decode(decoded[b].softmax(-1).argmax(-1)))

            # Break out of these loops if we are validating.
            if train and not train_on_real_data:
                break
        if train and not train_on_real_data:
            break
        if train_on_real_data:
            torch.save(conversionLayer, os.path.join(output_dir, "chkpt_real_%i.pt" % (epoch)))
