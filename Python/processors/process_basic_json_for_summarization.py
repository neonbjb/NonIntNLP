import json
import torch
from transformers import XLNetTokenizer

# Basic program which takes a input json file with a list of {'text','summary'} and outputs the stage-1 processed
# torch tensors needed by chunked_text_dataloader.
if __name__ == "__main__":
    with open('input.json') as f:
        datas = json.load(f)

    tok = XLNetTokenizer.from_pretrained("xlnet-base-cased")

    output = []
    for data in datas:
        text = data["text"]
        text_enc = tok.encode(
            text, add_special_tokens=False, max_length=None, pad_to_max_length=False
        )
        title = data["summary"]
        # Insert the title as the second sentence, forcing the proper token types.
        title_enc = tok.encode(
            title, add_special_tokens=False, max_length=None, pad_to_max_length=False
        )
        # Push resultants to a simple list and return it
        output.append({
            "text": torch.tensor(text_enc, dtype=torch.long),
            "target": torch.tensor(title_enc, dtype=torch.long),
        })

    torch.save(output, "out.pt")
