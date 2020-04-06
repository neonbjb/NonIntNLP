from torch.utils.data import Dataset, DataLoader, Sampler
import torch
import transformers
import math
import numpy as np
import random

# This class implements a Dataset which is capable of feeding a model which expects a stream of
# text with a generative target.
#
# It is unique because it supports extremely long sequences by "chunking" those sequences into several
# parts. Data retrieved from this set should then be fed into the model one chunk at a time until all chunks
# are exhausted. The target text of the language model will be found in the last chunk, so you should perform a
# backward() pass on the loss from that pass.
#
# To facilitate this, this dataset sorts the data it returns by chunk size. It is strongly recommended that you
# do not randomize this data without batching first.
#
# Clients should use the get_dataloader() method to retrieve a dataloader from this class. This
# custom dataloader will guarantee that retrieved batches have the same internal chunk length.
#
# An example where this might be used is text summarization, where long text is fed in and
# the generative target is a summary of that text.
class ChunkedTextDataset(Dataset):
    # data_file=File path to pytorch pickle file with a list of dicts containing tokenized {"text", "target"}
    # tokenizer=huggingface-spec tokenizer used to tokenize data_file.
    # max_chunk_len=Sequence size per chunk. This minus `max_gen_len` is the space left for the actual text.
    # max_gen_len=A fixed upper cap for the sequence length of the generated text.
    # add_pads_to_target=When true, target will be padded to max_target_len and model will predict pads. When false,
    #                    model will predict random elements inside of the text for all max_target_len elements over true
    #                    target_len.
    # mask_limit=Max number of masks applied to input_ids_masked. If 0, all elements will be masked.
    def __init__(
        self,
        data_file: str,
        tokenizer: transformers.PreTrainedTokenizer,
        max_chunk_len=192,
        max_gen_len=64,
        add_pads_to_target=True,
        mask_limit=0
    ):
        self.tokenizer = tokenizer
        self.max_chunk_len = max_chunk_len
        self.max_gen_len = max_gen_len
        self.data_file = data_file
        self.raw_data = torch.load(self.data_file)
        self.raw_data_sorted = False
        self.add_pads_to_target = add_pads_to_target
        self.mask_limit = mask_limit

    def process_element(self, text, target):
        # Tokens represented as 1-hot tensors which will be reused later in this function.
        bos_token_tensor = torch.tensor([self.tokenizer.bos_token_id], dtype=torch.long)
        eos_token_tensor = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)
        sep_token_tensor = torch.tensor([self.tokenizer.sep_token_id], dtype=torch.long)

        with torch.no_grad():
            target = torch.cat([target, eos_token_tensor])
            target_len = target.shape[0]
            # Truncate off the end of the target if it is too long.
            if target_len > self.max_gen_len:
                target = target[: self.max_gen_len]
                # Keep the <eos> token post-trucation.
                target[-1] = self.tokenizer.eos_token_id
                target_len = self.max_gen_len

            if self.add_pads_to_target:
                # Also pad target to max length. This is done to force the model to predict the actual end of the sequence.
                pads_needed = self.max_gen_len - target_len
                if pads_needed > 0:
                    target_padding = torch.full((pads_needed,), self.tokenizer.pad_token_id, dtype=torch.long)
                    target = torch.cat([target, target_padding])
                target_len = self.max_gen_len

            # Straight up append the target to the text, with the special tokens in between.
            text_with_target = torch.cat([bos_token_tensor, text, sep_token_tensor, target])

            # Add masks instead of the target to get masked_text.
            mask_offset_from_end = 5  #  Force the mask to have an offset from the end of the target to give the model context.
            if self.mask_limit >= (target_len-mask_offset_from_end) or self.mask_limit == 0:
                target_masks = torch.full((target_len,), self.tokenizer.mask_token_id, dtype=torch.long)
            else:
                target_mask_portion = torch.full((self.mask_limit,), self.tokenizer.mask_token_id, dtype=torch.long)
                target_masks = torch.cat([target[:-self.mask_limit-mask_offset_from_end], target_mask_portion, target[-mask_offset_from_end:]])
            text_masked = torch.cat([bos_token_tensor, text, sep_token_tensor, target_masks])

            # Create attention_masks that'll go along with this tokenized text. Ignore the pads in the target because
            # we want the model to predict them.
            attention_mask = torch.ones(text_with_target.shape[0], dtype=torch.float)

            # We will chunk all inputs so that none exceed max_chunk_len, which will all be fed into the model
            # sequentially. Lets figure out what the total lengths are first.
            num_chunks = math.ceil(text_with_target.shape[0] / self.max_chunk_len)
            final_text_seq_len = num_chunks * self.max_chunk_len

            # Before we can feed text_with_target into torch.chunk, it needs to be an exact multiple of text_len_per_chunk -
            # which is final_text_seq_len calculated above. Pad the text_with_target to accomplish this.
            padding_needed = final_text_seq_len - text_with_target.shape[0]
            padding_tensor = torch.full(
                (padding_needed,),
                fill_value=self.tokenizer.pad_token_id,
                dtype=torch.long,
            )
            att_padding_tensor = torch.zeros(padding_needed, dtype=torch.float)
            text_with_target = torch.cat([padding_tensor, text_with_target], dim=0)
            text_masked = torch.cat([padding_tensor, text_masked], dim=0)
            attention_mask = torch.cat([att_padding_tensor, attention_mask], dim=0)

            # The permutation mask is an eye across each chunk. The exception is the target text, which will be masked
            # sequentially to induce the model to predict one token at a time.
            perm_mask = torch.eye(
                self.max_chunk_len, dtype=torch.float
            ).repeat((num_chunks, 1))
            # Mask targets sequentially for targets.
            for t_index in range(1,target_len+1):
                perm_mask[-t_index][-t_index:] = 1.0
            # Mask all targets for all text.
            for t_index in range(target_len+1, self.max_chunk_len+1):
                perm_mask[-t_index][-target_len:] = 1.0

            # Build target mappings and labels. These are only included for use on the last chunk.
            target_mapping = torch.zeros((self.max_gen_len, self.max_chunk_len), dtype=torch.float)
            labels = torch.empty((self.max_gen_len,), dtype=torch.long)
            remaining_targets = self.max_gen_len - target_len
            masked_text_perm = np.random.permutation(self.max_chunk_len - target_len)
            tar_map_iter = 0
            for i in range(remaining_targets):
                target_mapping[tar_map_iter][masked_text_perm[i]] = 1
                labels[tar_map_iter] = text_with_target[masked_text_perm[i] - self.max_chunk_len]
                tar_map_iter += 1
            for i in range(target_len):
                if tar_map_iter >= 80:
                    print("error incoming. %i %i %i %i" % (i, target_len, tar_map_iter, remaining_targets))
                target_mapping[tar_map_iter][i - target_len] = 1
                labels[tar_map_iter] = text_with_target[i - target_len]
                tar_map_iter += 1

            chunked_text = torch.chunk(text_with_target, chunks=num_chunks)
            chunked_masked_text = torch.chunk(text_masked, chunks=num_chunks)
            chunked_attention = torch.chunk(attention_mask, chunks=num_chunks)
            chunked_perm_mask = torch.chunk(perm_mask, chunks=num_chunks, dim=0)

            result = {
                "input_ids": chunked_text,
                "input_ids_masked": chunked_masked_text,
                "attention_masks": chunked_attention,
                "permutation_masks": chunked_perm_mask,
                "target_mapping": target_mapping,
                "labels": labels,
            }
            return result

    def lazy_init(self):
        # Lazy load only when actually needed for performance.
        if not self.raw_data_sorted:
            self.raw_data.sort(key=lambda x: x["text"].shape[0])

    def num_chunks_for_index(self, i):
        self.lazy_init()
        text = self.raw_data[i]["text"]
        with torch.no_grad():
            return math.ceil(
                (text.shape[0] + self.max_gen_len + 3) / self.max_chunk_len
            )

    def __getitem__(self, index):
        self.lazy_init()
        return self.process_element(
            self.raw_data[index]["text"], self.raw_data[index]["target"]
        )

    def __len__(self):
        return len(self.raw_data)

    # Returns a batched dataloader for this dataset. See the documentation for ChunkedTextBatchSampler below for some
    # caveats on this.
    def get_dataloader(self, batch_sz: int, random=True, num_workers=1):
        return DataLoader(
            self,
            batch_sampler=ChunkedTextBatchSampler(self, batch_sz),
            shuffle=False,
            num_workers=num_workers,
        )


# This sampler will only return batches with the same chunk size.
#
# In order to achieve random=True with the above restriction, it "cheats". It does this by selecting a random permutation
# that "should" achieve coverage across the dataset when combined with the batch size, but almost certainly some elements
# will get skipped.
class ChunkedTextBatchSampler(Sampler):
    def __init__(
        self,
        chunked_text_set: ChunkedTextDataset,
        batch_sz: int,
        drop_last=True,
        random=True,
    ):
        self.chunked_text_set = chunked_text_set
        self.batch_sz = batch_sz
        self.drop_last = drop_last
        self.random = random

    def __iter__(self):
        batch = []
        batch_chunk_sz = 0

        if self.random:
            # minus batch_sz because we don't want to start from any element we cannot finish from.
            permutation = np.random.permutation(
                len(self.chunked_text_set) - self.batch_sz
            )
            yielded = 0
            for pidx in permutation:
                if yielded == len(self):
                    break
                batch = []
                chunk_sz = self.chunked_text_set.num_chunks_for_index(pidx)
                for b in range(self.batch_sz):
                    if chunk_sz != self.chunked_text_set.num_chunks_for_index(b + pidx):
                        break
                    batch.append(b + pidx)
                if len(batch) == self.batch_sz:
                    yielded += 1
                    yield batch

        else:
            for idx in range(len(self.chunked_text_set)):
                chunk_sz_idx = self.chunked_text_set.num_chunks_for_index(idx)
                if chunk_sz_idx != batch_chunk_sz:
                    batch.clear()
                    batch_chunk_sz = chunk_sz_idx
                batch.append(idx)
                if len(batch) == self.batch_sz:
                    yield batch
                    batch = []
                if len(batch) > 0 and not self.drop_last:
                    yield batch

    def __len__(self):
        # This is possibly (likely) not accurate, but it shouldn't matter.
        return int(len(self.chunked_text_set) / self.batch_sz)


###########################################################################################
# That's it for the functional logic. Everything else in this file is for testing
# the functionality of above. Included are some simple tests as well as some code you can
# use to see what the output of this dataloader would be against a given file.
###########################################################################################


def helpful_print_batch(batch, tokenizer):
    for b in range(batch["input_ids"][0].shape[0]):
        print("$$$$$$$$$$$$$$$  BATCH %i  $$$$$$$$$$$$$" % (b))
        batch_sz, chk_len = batch["input_ids"][0].shape
        chk_sz = len(batch["input_ids"])
        chk_it = 0
        target_len = batch["labels"].shape[-1]
        for input_ids, iim, att_mask, perm_mask in zip(
            batch["input_ids"],
            batch["input_ids_masked"],
            batch["attention_masks"],
            batch["permutation_masks"]
        ):
            print("************CHUNK %i/%i***********" % (chk_it + 1, chk_sz))
            chk_it += 1
            print("chunk inputs=%s" % (tokenizer.decode(input_ids[b])))
            print("masked inputs=%s" % (tokenizer.decode(iim[b])))
            print("***********************************")
            print(
                ">>>>>>>>>>ATTENTION MASK VIEW. SHOULD SEE NO PADS (only <unk>) IN TEXT."
            )
            masked = input_ids[b] * att_mask[b]
            print(tokenizer.decode(masked))
        print(
            ">>>>>>>>>>TARGET VIEW. FIRST IS PERM MASK VIEW FOR EACH TARGET IN LAST CHUNK.  SECOND LABELS. THIRD IS TARGET RECOMPILED FROM TARGET MAPPINGS."
        )
        labels_fixed = batch["labels"][b].tolist()

        last_chunk_ids = batch["input_ids"][-1][b]
        last_chunk_perms = batch["permutation_masks"][-1][b]
        last_target_mapping = batch["target_mapping"][b]
        recompiled_target = []
        for t in range(last_target_mapping.shape[0]):
            # Find the target_mapping inside of input_ids.
            target_index = None
            for i in range(chk_len):
                if last_target_mapping[t][i] == 1.0:
                    target_index = i
                    break
            if target_index is not None:
                recompiled_target.append(last_chunk_ids[target_index])
                target_word = tokenizer.decode([last_chunk_ids[target_index]])

            # perm_mask is 1 where masked, 0 where not masked. we need to invert that to make the masking easy.
            perm_mask_inverted = (~(last_chunk_perms[target_index].bool())).long()
            masked = last_chunk_ids * perm_mask_inverted

            print("[%i] %s: `%s`" % (target_index, target_word, tokenizer.decode(masked)))
        print(tokenizer.decode(labels_fixed))
        print(tokenizer.decode(recompiled_target))
        print("***********************************")


# Test the tokenizer.
# TBD..
def perform_tests(batch, chk_sz):
    batchsz = batch["input_ids"][0].shape[0]
    for c in range(chk_sz):
        print("chunk %i" % (c))
        for b in range(batchsz):
            labels = batch["labels"][b]
            if random.uniform(0, 1) < 0.9:
                # Just skip most of these. We only need a few of them for testing.
                continue
            print("batch element %i" % (b))
            inputs = batch["input_ids"][c][b]
            mask = batch["attention_masks"][c][b]
            perm_mask = batch["permutation_masks"][c][b]
            Check(perm_mask.shape).equals((len(inputs), len(inputs)))


def test_against_real_file(test_file, tokenizer):
    batchsz = 16
    dataset = ChunkedTextDataset(
        data_file=test_file,
        tokenizer=tokenizer,
        max_chunk_len=256,
        max_gen_len=80,
        add_pads_to_target=False,
        mask_limit=6
    )
    loader = dataset.get_dataloader(batch_sz=batchsz)

    _b_n = 0
    for batch in loader:
        chk_sz = len(batch["labels"])

        print("Processing batch %i/%i chunk_sz = %i" % (_b_n, len(loader), chk_sz))
        _b_n += 1

        helpful_print_batch(batch, tokenizer)
        perform_tests(batch, chk_sz)

if __name__ == "__main__":
    from fluentcheck import Check

    tokenizer = transformers.XLNetTokenizer.from_pretrained("xlnet-base-cased")

    # Provided for testing.
    test_file = (
        "C:\\Users\\jbetk\\Documents\\data\\ml\\xsum\\xsum-extracts-from-downloads\\outputs\\val.pt"
    )
    test_against_real_file(test_file, tokenizer)
