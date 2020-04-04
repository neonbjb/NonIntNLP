import torch
import transformers

# This dataset reads data from a stage 1 processor which converts a large set of wikipedia data into individual
# sentences which are pre-processed by a tokenizer.
#
# It recombines those sentences such that the tensor outputs are exactly the specified seq_sq. When pulling from the
# dataset, indices skip the sentences within the dataset by 3, which avoids getting the same sentence over and over.
# Realistically, it is expected that you shuffle your pulls to maximize entropy.
class WikiMaskDataset(torch.utils.data.Dataset):
    def __init__(self, datapath=str, tokenizer=transformers.tokenization_utils.PreTrainedTokenizer, seq_sz=int, num_elements_masked=int):
        self.raw_data = torch.load(datapath)
        self.tokenizer = tokenizer
        self.seq_sz = seq_sz
        self.num_elements_masked = num_elements_masked

    def __getitem__(self, index):
        with torch.no_grad():
            index *= 3
            tensors = []
            # Tokenize an empty string to append special characters
            tensors.append(torch.tensor(self.tokenizer.encode("", add_special_tokens=True), dtype=torch.long))
            tensor_len = tensors[-1].shape[0]
            while tensor_len <= self.seq_sz:
                tensors.append(torch.tensor(self.raw_data[index], dtype=torch.long))
                tensor_len += tensors[-1].shape[0]
                index += 1
            item = torch.cat(tensors)

            # Limit item to a max length.
            item = item[:self.seq_sz]

            # Now handle masking.
            item_masked = item.clone().detach()
            item_masked[-self.num_elements_masked:] = self.tokenizer.mask_token_id

        return {"input_ids": item, "input_ids_masked": item_masked}

    def __len__(self):
        return int(len(self.raw_data) / 3)

