import os
import torch
import transformers
import random
from dataloaders.wiki_mask_dataset import WikiMaskDataset

# Root directory for dataset
dataroot = "C:\\Users\\jbetk\\Documents\\data\\ml\\wiki\\processed"
# Output directory for model checkpoints.
output_dir = "C:\\Users\\jbetk\\Documents\\data\\ml\\saved_models\\ganformer\\test_chkpt"
device_name = "cuda"

# new stuff
model_name = "albert-large-v2"
seq_sz = 256
#
num_graded_tokens = 10
num_masked_tokens = 5
# This is basically just the batch size of this interpreter.
batch_size = 4
num_to_generate = 20000
discriminator = False
# Decide which device we want to run on
device = torch.device(device_name)

tokenizer = transformers.AlbertTokenizer.from_pretrained(
    model_name
)
dataset = WikiMaskDataset(
    os.path.join(dataroot, "train.pt"),
    tokenizer,
    seq_sz,
    num_elements_masked=num_masked_tokens
)
configG = transformers.AlbertConfig.from_pretrained(
    model_name
)
configG.output_hidden_states = 1
configD = transformers.AlbertConfig.from_pretrained(
    model_name
)
configD.num_labels = 2

# Create the models
generator = transformers.AlbertForMaskedLM.from_pretrained(model_name, config=configG).to(device)
if discriminator:
    discriminator = transformers.AlbertForSequenceClassification.from_pretrained(model_name, config=configD).to(device)

random.seed(12345)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True)

# Albert has a special layer between the hidden states and the embeddings. Consider transferring these
# states across models rather than the embedding states.
def generator_hidden_states_to_embedding(hidden_states):
    hidden_states = generator.predictions.dense(hidden_states)
    hidden_states = generator.predictions.activation(hidden_states)
    hidden_states = generator.predictions.LayerNorm(hidden_states)
    return hidden_states

generated = 0
generated_list = []
while generated < num_to_generate:
    with torch.no_grad():
        batch = next(dataloader.__iter__())
        g_inputs = {
            "input_ids": batch["input_ids_masked"].to(device)
        }
        # Feed forward and process results.
        logitsG, hidden = generator(**g_inputs)

        _embeddings = generator.get_input_embeddings().forward(batch["input_ids"].to(device))
        final_hidden_state = generator_hidden_states_to_embedding(hidden[-1])
        disc_embeddings = torch.cat([_embeddings[:, :-num_graded_tokens, :], final_hidden_state[:, -num_graded_tokens:, :]], dim=1)
        d_inputs = {
            "inputs_embeds": disc_embeddings
        }

        tokens = logitsG.softmax(dim=-1).argmax(dim=-1)
        for b in range(batch_size):
            print("Input: %s" % tokenizer.decode(batch["input_ids_masked"][b]))
            print("Output: %s" % tokenizer.decode(tokens[b]))

            generated_list.append({
                "input_ids": batch["input_ids"][b].cpu().detach(),
                "generated_ids": tokens[b].cpu().detach(),
                "embeddings": _embeddings[b].cpu().detach(),
                "gen_embeddings": disc_embeddings.cpu().detach()
            })
            generated += 1

        if discriminator:
            logitsD = discriminator(**d_inputs)
            print("Discriminator losses:" + str(logitsD[0].softmax(-1).to("cpu").numpy()))

torch.save(generated_list, os.path.join(output_dir, "generated_data.pt"))
