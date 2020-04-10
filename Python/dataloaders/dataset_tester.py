import transformers
import torch
import wandb
import time
import os
import argparse
import json
from dataloaders.chunked_text_dataloader import ChunkedTextDataset
from tqdm import tqdm
from apex import amp

if __name__ == '__main__':
    tokenizer = transformers.XLNetTokenizer.from_pretrained("xlnet-base-cased")
    dataset = ChunkedTextDataset(
        "C:\\Users\\jbetk\\Documents\\data\\ml\\xsum\\xsum-extracts-from-downloads\\outputs\\train.pt",
        tokenizer,
        256,
        80,
        add_pads_to_target=True
    )

    # num_workers > 1
    loader = dataset.get_dataloader(4, num_workers=1)
    _epoch_iterator = tqdm(
        loader, desc="Train Iteration")

    for batch in _epoch_iterator:
        print(tokenizer.decode(batch['input_ids'][0][0]))