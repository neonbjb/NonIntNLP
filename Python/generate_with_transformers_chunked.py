import torch
import transformers
from dataloaders.chunked_text_dataloader import ChunkedTextDataset
import os
import json
import random
import argparse

parser = argparse.ArgumentParser(
    description="Train an auto-regressive transformer model."
)
parser.add_argument(
    "--model_dir",
    type=str,
    required=True,
    help="Where to load the saved transformers model from. This is also where the output is written to.",
)
parser.add_argument(
    "--data_file",
    type=str,
    required=True,
    help="The test data file to use for generation. Should be loadable by chunked_text_dataloader.",
)
parser.add_argument(
    "--number_to_generate",
    type=int,
    default=5,
    help="The number of test_data rows to pull and generate from.",
)
parser.add_argument(
    "--device", type=str, default="cuda", help="The device to use for generation.",
)
parser.add_argument(
    "--do_sampling", type=bool, default=False, help="The device to use for generation.",
)
args = parser.parse_args()

number_to_generate = args.number_to_generate
device_str = args.device
model_dir = args.model_dir
data_file = args.data_file
do_sampling = args.do_sampling

# Load the chunk config. This is saved alongside the model if it was trained using train_xlnet_lang_modeler
# and saves some of the configuration options used to train the model which are also useful for generation.
with open(os.path.join(model_dir, "chunk_config.json"), "r") as chunk_cfg:
    chunk_config = json.load(chunk_cfg)
chunk_seq_len = chunk_config["max_seq_len"]
target_pred_max_len = chunk_config["predict_len"]

# Load the tokenizer, config and model.
tokenizer = transformers.XLNetTokenizer.from_pretrained(chunk_config["model_name"])
config = transformers.XLNetConfig.from_pretrained(chunk_config["model_name"])
config.mem_len = chunk_config["mem_len"]
model = transformers.XLNetLMHeadModel.from_pretrained(model_dir, config=config)
model.eval()

# Load the dataset for the file specified above. This class is the same one used during training to perform chunking
# and some preprocessing.
test_set = ChunkedTextDataset(
    data_file,
    tokenizer,
    chunk_config["max_seq_len"],
    chunk_config["predict_len"],
    pad_left=True,
)

random.seed(12345)
device = torch.device(device_str)
model.to(device)
results = {"meta": chunk_config, "results": []}
for i in range(number_to_generate):
    chunked_data = test_set[random.randint(0, len(test_set) - 1)]
    full_text = []

    # Start by establishing mems state - achieved by doing forward passes on all chunks but the last one.
    mems = None
    num_chunks = len(chunked_data["input_ids"])
    for c in range(num_chunks - 1):
        full_text.extend(chunked_data["input_ids"][c].tolist())

        model_inputs = {
            "input_ids": chunked_data["input_ids"][c].unsqueeze(0).to(device),
            "attention_mask": chunked_data["attention_masks"][c]
            .unsqueeze(0)
            .to(device),
            "perm_mask": chunked_data["permutation_masks"][c].unsqueeze(0).to(device),
            "target_mapping": chunked_data["target_mappings"][c]
            .unsqueeze(0)
            .to(device),
        }
        if mems is not None:
            model_inputs["mems"] = mems
        with torch.no_grad():
            logits, mems = model.forward(**model_inputs)

    # Now get the input IDs minus the target* for the last chunk. This will serve as the "prompt" for the model.
    text_len = chunk_seq_len - target_pred_max_len
    prompt_inputs = chunked_data["input_ids"][-1][0:text_len]
    prompt_inputs = prompt_inputs.to(device)
    full_text.extend(prompt_inputs)
    prompt_inputs = prompt_inputs.unsqueeze(dim=0)  # generate() expects batched inputs.

    # Use the transformers generate function to do the actual generation now.
    if do_sampling:
        genned_results = model.generate(
            prompt_inputs,
            max_length=chunk_seq_len,
            do_sample=True,
            num_beams=7,
            temperature=0.7,
            top_k=0,
            top_p=0.9,
            repetition_penalty=5,
            eos_token_ids=tokenizer.eos_token_id,
            num_return_sequences=5,
            mems=mems,
        )
    else:
        genned_results = model.generate(
            prompt_inputs,
            max_length=chunk_seq_len,
            do_sample=False,
            num_beams=12,
            repetition_penalty=5,
            eos_token_ids=tokenizer.eos_token_id,
            num_return_sequences=2,
            mems=mems,
        )

    # Append results here.
    seqs, _ = genned_results.shape
    genned_texts = []
    prompt = tokenizer.decode(prompt_inputs[0])
    print(
        "\n------------------------------------------------------------------------------"
    )
    print("PROMPT: `%s`" % (prompt))
    for s in range(seqs):
        genned_texts.append(tokenizer.decode(genned_results[s][text_len:]))
        print("GENERATED: `%s`" % (genned_texts[-1]))
    result = {
        "prompt": tokenizer.decode(full_text),
        "generated": genned_texts,
        "actual": tokenizer.decode(chunked_data["labels"]),
    }
    print(
        "------------------------------------------------------------------------------"
    )
    results["results"].append(result)

# Record the results.
with open(model_dir + "/gen_results.json", "w", encoding="utf-8") as result_file:
    json.dump(results, result_file, sort_keys=True, indent=4)
