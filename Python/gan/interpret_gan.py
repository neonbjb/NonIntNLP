import os
import torch
import transformers
import random
from dataloaders.chunked_text_dataloader import ChunkedTextDataset

# Root directory for dataset
dataroot = "C:\\Users\\jbetk\\Documents\\data\\ml\\xsum\\xsum-extracts-from-downloads\\outputs"
output_dir = "C:\\Users\\jbetk\\Documents\\data\\ml\\saved_models\\ganformer"
device_name = "cuda"

# new stuff
model_name = "xlnet-base-cased"
seq_sz = 256
max_predict_sz = 58
mem_len = 768
num_to_generate = 5
# Decide which device we want to run on
device = torch.device(device_name)

tokenizer = transformers.XLNetTokenizer.from_pretrained(model_name)
configG = transformers.XLNetConfig.from_pretrained(model_name)
configG.mem_len = mem_len
configG.output_hidden_states = 1
generator = transformers.XLNetLMHeadModel.from_pretrained(os.path.join(output_dir, "test_chkpt/generator"), config=configG)
generator.to(device)

configD = transformers.XLNetConfig.from_pretrained(model_name)
configD.mem_len = mem_len
configD.output_hidden_states = 1
configD.num_labels = 1
discriminator = transformers.XLNetForSequenceClassification.from_pretrained(os.path.join(output_dir, "test_chkpt/discriminator"), config=configD)
discriminator.to(device)

dataset = ChunkedTextDataset(
    os.path.join(dataroot, "test.pt"),
    tokenizer,
    seq_sz,
    max_predict_sz,
    pad_left=True,
)
random.seed(12345)
loader = dataset.get_dataloader(batch_sz=num_to_generate, random=True, num_workers=0)
batch = next(loader.__iter__())
num_chunks = len(batch["input_ids"])

with torch.no_grad():

    # Establish mems.
    def computeMemsForModel(gen, disc):
        memsG, memsD = None, None
        for c in range(num_chunks - 1):
            inputs = {
                "input_ids": batch["input_ids"][c].to(device),
                "attention_mask": batch["attention_masks"][c].to(device),
            }
            if memsG is not None:
                inputs["mems"] = memsG
            with torch.no_grad():
                logits, memsG, hidden = gen.forward(**inputs)
                inputsD = {
                    "inputs_embeds": hidden[-1],
                    "attention_mask": batch["attention_masks"][c].to(device),
                }
                if memsD is not None:
                    inputs["mems"] = memsD
                logits, memsD, hidden = disc.forward(**inputs)

        return memsG, memsD


    # Compute the mems for both the discriminator and the generator.
    memsG, memsD = computeMemsForModel(generator, discriminator)

    # Generate noise to apply across the <masked> tokens.
    g_inputs = {
        "input_ids": batch["input_ids_masked"][-1].to(device),
        "attention_mask": batch["attention_masks"][-1].to(device),
        "mems": memsG
    }

    # Feed forward and process results.
    logits, memsG, hidden = generator(**g_inputs)

    d_inputs = {
        "inputs_embeds": hidden[-1],
        "attention_mask": batch["attention_masks"][-1].to(device),
        "mems": memsD
    }
    logitsD, memsD, hidden = discriminator(**d_inputs)

    tokens = logits.softmax(dim=-1).argmax(dim=-1)
    for b in range(num_to_generate):
        print("Input: %s" % tokenizer.decode(batch["input_ids_masked"][-1][b]))
        print("Output: %s" % tokenizer.decode(tokens[b]))
    print("Discriminator losses:" + str(logitsD.to("cpu").numpy()))