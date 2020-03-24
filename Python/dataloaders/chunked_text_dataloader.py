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
    # pad_left=Whether the padding should go on the left or the right.
    def __init__(
        self,
        data_file: str,
        tokenizer: transformers.PreTrainedTokenizer,
        max_chunk_len=192,
        max_gen_len=64,
        pad_left=False,
    ):
        self.tokenizer = tokenizer
        self.max_chunk_len = max_chunk_len
        self.max_gen_len = max_gen_len
        self.pad_left = pad_left
        self.raw_data = torch.load(data_file)
        self.raw_data.sort(key=lambda x: x["text"].shape[0])

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

            # The target will be the end of the sequence, so append that token onto it.
            # Also pad target to max length. This is done to force the model to predict the actual end of the sequence.
            pads_needed = self.max_gen_len - target_len
            if pads_needed > 0:
                target_padding = torch.full((pads_needed,), self.tokenizer.pad_token_id, dtype=torch.long)
                target = torch.cat([target, target_padding])
            target_len = self.max_gen_len

            # Straight up append the target to the text, with the special tokens in between.
            text = torch.cat([bos_token_tensor, text, sep_token_tensor, target])

            # Create attention_masks that'll go along with this tokenized text. Ignore the pads in the target because
            # we want the model to predict them.
            attention_mask = torch.ones(text.shape[0], dtype=torch.float)

            # The permutation mask is all 0s for the text; all tokens attend to each other. For the target, our goal is
            # sequential generation, so the permutation will also be sequential. We won't let the text perform any
            # attention on the title.
            text_perm_mask = torch.zeros(
                (self.max_chunk_len, text.shape[0] - target_len), dtype=torch.float
            )
            target_perm_mask = torch.ones(
                (self.max_chunk_len, target_len), dtype=torch.float
            )
            for t_index in range(target_len):
                target_perm_mask[-t_index][-target_len:-t_index] = 0.0
            perm_mask = torch.cat([text_perm_mask, target_perm_mask], dim=-1)

            # Build target mappings. These are identical for all chunks - the target is just an eye tensor
            # up to target_len that is shifted to the end of the text sequence, with zeros placed everywhere.
            target_mapping = torch.eye(target_len, dtype=torch.float)
            target_mapping_shift = torch.zeros((target_len, text.shape[0] - target_len), dtype=torch.float)
            target_mapping = torch.cat([target_mapping_shift, target_mapping], dim=-1)

            # We will chunk all inputs so that none exceed max_chunk_len, which will all be fed into the model
            # sequentially. Lets figure out what the total lengths are first.
            num_chunks = math.ceil(text.shape[0] / self.max_chunk_len)
            final_text_seq_len = num_chunks * self.max_chunk_len

            # Before we can feed text into torch.chunk, it needs to be an exact multiple of text_len_per_chunk -
            # which is final_text_seq_len calculated above. Pad the text to accomplish this.
            padding_needed = final_text_seq_len - text.shape[0]
            padding_tensor = torch.full(
                (padding_needed,),
                fill_value=self.tokenizer.pad_token_id,
                dtype=torch.long,
            )
            att_padding_tensor = torch.zeros(padding_needed, dtype=torch.float)
            perm_padding_tensor = torch.zeros((self.max_chunk_len, padding_needed), dtype=torch.float)
            tar_map_padding_tensor = torch.zeros((target_len, padding_needed), dtype=torch.float)
            if self.pad_left:
                text = torch.cat([padding_tensor, text], dim=0)
                attention_mask = torch.cat([att_padding_tensor, attention_mask], dim=0)
                perm_mask = torch.cat([perm_padding_tensor, perm_mask], dim=-1)
                target_mapping = torch.cat([tar_map_padding_tensor, target_mapping], dim=-1)
            else:
                text = torch.cat([text, padding_tensor], dim=0)
                attention_mask = torch.cat([attention_mask, att_padding_tensor], dim=0)
                perm_mask = torch.cat([perm_mask, perm_padding_tensor], dim=-1)
                target_mapping = torch.cat([target_mapping, tar_map_padding_tensor], dim=-1)

            chunked_text = torch.chunk(text, chunks=num_chunks)
            chunked_attention = torch.chunk(attention_mask, chunks=num_chunks)
            chunked_perm_mask = torch.chunk(perm_mask, chunks=num_chunks, dim=-1)
            chunked_target_mapping = torch.chunk(target_mapping, chunks=num_chunks, dim=-1)
            labels = target

            result = {
                "input_ids": chunked_text,
                "attention_masks": chunked_attention,
                "permutation_masks": chunked_perm_mask,
                "target_mappings": chunked_target_mapping,
                "labels": labels,
            }
            return result

    def num_chunks_for_index(self, i):
        text = self.raw_data[i]["text"]
        with torch.no_grad():
            return math.ceil(
                (text.shape[0] + self.max_gen_len + 3) / self.max_chunk_len
            )

    def __getitem__(self, index):
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
        for input_ids, att_mask, perm_mask, target_mapping in zip(
            batch["input_ids"],
            batch["attention_masks"],
            batch["permutation_masks"],
            batch["target_mappings"],
        ):
            print("************CHUNK %i/%i***********" % (chk_it + 1, chk_sz))
            chk_it += 1
            print("chunk inputs=%s" % (tokenizer.decode(input_ids[b])))
            print("***********************************")
            print(
                ">>>>>>>>>>ATTENTION MASK VIEW. SHOULD SEE NO PADS (only <unk>) IN TEXT."
            )
            masked = input_ids[b] * att_mask[b]
            print(tokenizer.decode(masked))
        print(
            ">>>>>>>>>>TARGET VIEW. FIRST LABELS. SECOND IS PERM MASK VIEW FOR EACH TARGET IN LAST CHUNK. THIRD IS TARGET RECOMPILED FROM TARGET MAPPINGS."
        )
        labels_fixed = batch["labels"][b].tolist()
        print(tokenizer.decode(labels_fixed))

        last_chunk_ids = batch["input_ids"][-1][b]
        last_chunk_perms = batch["permutation_masks"][-1][b]
        last_target_mappings = batch["target_mappings"][-1][b]
        recompiled_target = []
        true_counter = -5
        for t in range(-target_len - 5, -1): # -5 to grab the last few true input IDs to check those as well.
            # perm_mask is 1 where masked, 0 where not masked. we need to invert that to make the masking easy.
            perm_mask_list = last_chunk_perms[t].tolist()
            perm_mask_list = [1.0 if ele == 0 else 0.0 for ele in perm_mask_list]
            perm_mask_inverted = torch.tensor(perm_mask_list, dtype=torch.float)
            masked = last_chunk_ids * perm_mask_inverted
            print(tokenizer.decode(masked))

            if true_counter >= 0:
                # Find the target_mapping inside of input_ids.
                target_index = None
                for i in range(chk_len):
                    if last_target_mappings[true_counter][i] == 1.0:
                        target_index = i
                        break
                if target_index is not None:
                    recompiled_target.append(last_chunk_ids[target_index])

            true_counter += 1
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
            Check(perm_mask.shape).equals((len(ii), len(ii)))


def test_against_real_file(test_file, tokenizer):
    batchsz = 16
    dataset = ChunkedTextDataset(
        data_file=test_file,
        tokenizer=tokenizer,
        max_chunk_len=256,
        max_gen_len=32,
        pad_left=True,
    )
    loader = dataset.get_dataloader(batch_sz=batchsz)

    _b_n = 0
    for batch in loader:
        chk_sz = len(batch["labels"])

        print("Processing batch %i/%i chunk_sz = %i" % (_b_n, len(loader), chk_sz))
        _b_n += 1

        helpful_print_batch(batch, tokenizer)
        perform_tests(batch, chk_sz)


# This test applies the dataset rules against some raw text, then attempts to print out the results as the model would
# see them.
def test_against_test_set(tokenizer):
    # Check the actual conversions.
    def test_enc(str):
        return torch.tensor(
            tokenizer.encode(str, add_special_tokens=False), dtype=torch.long
        )

    test_set = [
        {
            "text": test_enc(
                "President Donald Trump’s new European travel restrictions have a convenient side effect"
            ),
            "target": test_enc("trump is an asshat"),
        },
        {
            "text": test_enc(
                """
                Trump is already under fire for visiting his properties in both countries as president, leading to U.S. taxpayer money being spent at his own firms. The president has been saddled with lawsuits and investigations throughout his term alleging that he’s violating the Constitution’s emoluments clause by accepting taxpayer money other than his salary.
                The U.S. government proclamation initiating the ban targets 26 European countries that comprise a visa-free travel zone known as the Schengen Area.
                The United Kingdom, which is home to Trump Turnberry and Trump International Golf Links, and Ireland, which is home to another Trump-branded hotel and golf course at Doonbeg, do not participate in the Schengen Area. Bulgaria, Croatia and Romania are also not part of the Schengen Area. All three of the resorts are struggling financially.
                Ireland’s prime minister, Leo Varadkar, is scheduled to meet Trump at the White House on Thursday in one of the few events related to St. Patrick’s Day that has not been canceled due to coronavirus concerns.
                The administration’s European travel proclamation notes that “the Schengen Area has exported 201 COVID-19 cases to 53 countries. Moreover, the free flow of people between the Schengen Area countries makes the task of managing the spread of the virus difficult.”
                Trump’s European travel ban comes with several other loopholes.
                Though they are subject to border checks on arrival, residents of the 26 Schengen Area countries are also free to live and work in the United Kingdom, meaning they could fly to the United States from a British airport as long as they hadn't spent time within the Schengen countries in the last 14 days.
                EU leaders condemned Trump's move on Thursday, and disputed the president's criticism of Europe's handling of the crisis.
                “The Coronavirus is a global crisis, not limited to any continent and it requires cooperation rather than unilateral action,” European Commission President Ursula von der Leyen and European Council President Charles Michel said in a joint statement.
                """
            ),
            "target": test_enc("trump is still an asshat"),
        },
    ]
    torch.save(test_set, "test.pt")
    dataset = ChunkedTextDataset(data_file="test.pt", tokenizer=tokenizer)
    loader = dataset.get_dataloader(batch_sz=1)
    batch_it = 0
    for batch in loader:
        print("##########BATCH %i#############" % (batch_it))
        batch_it += 1
        helpful_print_batch(batch, tokenizer)


if __name__ == "__main__":
    from fluentcheck import Check

    tokenizer = transformers.XLNetTokenizer.from_pretrained("xlnet-base-cased")

    # Provided for testing.
    test_file = (
        "C:\\Users\\jbetk\\Documents\\data\\ml\\title_prediction\\outputs\\val.pt"
    )
    test_against_real_file(test_file, tokenizer)
    # test_against_test_set(tokenizer)
