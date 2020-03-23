from torch.utils.data import Dataset, DataLoader, Sampler
import torch
import transformers
import math
from typing import Tuple
import numpy as np
import random

# This class implements a Dataset which is capable of feeding a model which expects a stream of
# text with a generative target. It also (optionally) performs random masking on the target
# for the purposes of doing masked pre-training.
#
# It is unique because it supports extremely long sequences by "chunking" those sequences into several
# parts. Data retrieved from this set should then be fed into the model one chunk at a time until all chunks
# are exhausted, then more data should be fetched from the set.
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
    # mask_target_percentage=The proportion of target tokens to mask.
    # mask_all_percentage=The proportion of <all> tokens to mask.
    # force_max_len_gen=If true, target text will be padded to <max_gen_len> in every input sequence and token type IDs
    #                   will be generated.
    # pad_left=Whether the padding should go on the left or the right.
    # includes_classification=When true, the dataloader will extract a classification token from the data file. This can be used to train critic models.
    # target_mapping_labels=When true, the dataset will output "target_mapping" and the "labels" output will be tied to that, rather than the entire sequence.
    def __init__(
        self,
        data_file: str,
        tokenizer: transformers.PreTrainedTokenizer,
        max_chunk_len=192,
        max_gen_len=64,
        mask_target_percentage=0.3,
        mask_all_percentage=0.1,
        force_max_len_gen=False,
        pad_left=False,
        includes_classification=False,
        target_mapping_labels=False,
    ):
        self.tokenizer = tokenizer
        self.max_chunk_len = max_chunk_len
        self.max_gen_len = max_gen_len
        self.mask_target_percentage = mask_target_percentage
        self.mask_all_percentage = mask_all_percentage
        self.pad_left = pad_left
        self.force_max_len_gen = force_max_len_gen
        self.raw_data = torch.load(data_file)
        self.raw_data.sort(key=lambda x: x["text"].shape[0])
        self.includes_classification = includes_classification
        self.target_mapping_labels = target_mapping_labels

        # force_max_len_gen is a constraint of target_mapping_labels. This is because the class that does batching
        # requires that all labels have the same length, and target_mapping_labels forces labels to be of length target_text.
        if target_mapping_labels:
            assert force_max_len_gen

    def process_element(self, text, target, classifier):
        # Tokens represented as 1-hot tensors which will be reused later in this function.
        bos_token_tensor = torch.tensor([self.tokenizer.bos_token_id], dtype=torch.long)
        eos_token_tensor = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)

        with torch.no_grad():
            # Targets and text always get <bos> and <eos> on them.
            target = torch.cat([bos_token_tensor, target, eos_token_tensor])
            text = torch.cat([bos_token_tensor, text, eos_token_tensor])

            target_len = target.shape[0]
            if target_len > self.max_gen_len:
                # Truncate.
                target = target[: self.max_gen_len]
                # Keep the <eos> token post-trucation.
                target[-1] = self.tokenizer.eos_token_id
                target_len = self.max_gen_len
            elif target_len < self.max_gen_len and self.force_max_len_gen:
                masks_to_fill_target = self.max_gen_len - target_len
                masks_tensor = torch.full(
                    (masks_to_fill_target,),
                    self.tokenizer.pad_token_id,
                    dtype=torch.long,
                )
                target = torch.cat([target, masks_tensor])
                target_len = self.max_gen_len

            # Create attention_masks that'll go along with this tokenized text.
            attention_mask = torch.ones(text.shape[0], dtype=torch.float)

            # We will chunk all inputs so that none exceed max_chunk_len, which will all be fed into the model
            # sequentially. Some set-up is necessary first.
            text_len_per_chunk = self.max_chunk_len - target_len
            num_chunks = math.ceil(text.shape[0] / text_len_per_chunk)
            final_text_seq_len = num_chunks * text_len_per_chunk

            # Before we can feed text into torch.chunk, it needs to be an exact multiple of text_len_per_chunk. This
            # will be accomplished by padding.
            padding_needed = final_text_seq_len - text.shape[0]
            padding_tensor = torch.full(
                (padding_needed,),
                fill_value=self.tokenizer.pad_token_id,
                dtype=torch.long,
            )
            att_padding_tensor = torch.zeros(padding_needed, dtype=torch.float)
            if self.pad_left:
                text = torch.cat([padding_tensor, text], dim=0)
                attention_mask = torch.cat([att_padding_tensor, attention_mask], dim=0)
            else:
                text = torch.cat([text, padding_tensor], dim=0)
                attention_mask = torch.cat([attention_mask, att_padding_tensor], dim=0)

            # The labels are easy to init.
            labels = torch.full(
                (final_text_seq_len,), fill_value=-100, dtype=torch.long
            )

            # The permutation mask is also easy - all 0s for the text; all tokens attend to each other.
            perm_mask = torch.zeros(
                (self.max_chunk_len, final_text_seq_len), dtype=torch.float
            )

            chunked_text = torch.chunk(text, chunks=num_chunks)
            chunked_attention = torch.chunk(attention_mask, chunks=num_chunks)
            chunked_labels = torch.chunk(labels, chunks=num_chunks)
            chunked_perm_mask = torch.chunk(perm_mask, chunks=num_chunks, dim=-1)

            # Create a permutation mask for the target. Our goal is sequential generation, so the permutation will
            # also be sequential. We won't let the text perform any attention on the title.
            target_permutation = torch.ones(
                (self.max_chunk_len, target_len), dtype=torch.float
            )
            for t_index in range(target_len):
                target_permutation[t_index][0:t_index] = 0.0

            # Build target mappings if needed. These are identical for all chunks - the target is just an eye tensor
            # up to target_len and zeros everywhere else.
            target_mapping = None
            if self.target_mapping_labels:
                target_mapping = torch.eye(target_len, self.max_chunk_len, dtype=torch.float)

            # Perform masking on the target if needed.
            target_masked = target.clone().detach()
            target_masked, label_append = self.perform_mask(
                target_masked,
                self.mask_target_percentage,
            )

            # If we are using target_mapping, than the labels will always get exactly the target and nothing else.
            if self.target_mapping_labels:
                label_append.copy_(target)

            # Now append the labels (and masks) per chunk
            input_ids = []
            input_ids_masked = []
            attention_mask = []
            labels = []
            permutation_mask = []
            target_mappings = []
            classifiers = None
            token_type_ids = None
            if self.force_max_len_gen:
                token_type_ids = []
            if self.includes_classification:
                classifiers = []
                encoded_classifier = torch.tensor(classifier, dtype=torch.float)

            # Now we're going to concatenate the per-chunk parts with the target parts.
            for c_text, c_att, c_lab, c_perm in zip(
                chunked_text, chunked_attention, chunked_labels, chunked_perm_mask
            ):
                input_ids.append(torch.cat([target, c_text,], dim=0,))
                attention_mask.append(
                    torch.cat(
                        [torch.ones(target_len, dtype=torch.float), c_att,], dim=0,
                    )
                )
                c_text_masked = torch.cat([target_masked, c_text,], dim=0,)
                c_perm_mask = torch.cat([target_permutation, c_perm], dim=-1,)

                if self.target_mapping_labels:
                    c_lab_full = label_append
                else:
                    # Without target_mapping, the labels is the full sequence. Presumably in this case we're using masking so do that too.
                    c_lab_full = torch.cat([label_append, c_lab,], dim=0,)
                    # Now we just have to perform full masking.
                    c_text_masked, c_lab_full = self.perform_mask(
                        c_text_masked, self.mask_all_percentage, c_lab_full, input_ids[-1]
                    )

                input_ids_masked.append(c_text_masked)
                labels.append(c_lab_full)
                permutation_mask.append(c_perm_mask)
                target_mappings.append(target_mapping)

                if self.force_max_len_gen:
                    token_type_ids.append(
                        torch.cat(
                            [
                                torch.zeros((target_len + 2,), dtype=torch.long),
                                torch.ones(
                                    (self.max_chunk_len - target_len - 2,),
                                    dtype=torch.long,
                                ),
                            ]
                        )
                    )
                if self.includes_classification:
                    classifiers.append(encoded_classifier)

            result = {
                "input_ids": input_ids,
                "input_ids_masked": input_ids_masked,
                "attention_masks": attention_mask,
                "labels": labels,
                "permutation_masks": permutation_mask,
                # This is included for debugging and testing
                "target_len": [
                    torch.tensor(target_len, dtype=torch.long)
                    for cs in range(num_chunks)
                ],
            }
            if token_type_ids is not None:
                result["token_type_ids"] = token_type_ids
            if self.includes_classification:
                result["classifiers"] = classifiers
            if self.target_mapping_labels:
                result["target_mappings"] = target_mappings
            return result

    def perform_mask(
        self,
        tensor: torch.Tensor,
        mask_percentage: float,
        prefilled_labels=None,
        truth_values=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        # "truth_values" are for when we are performing double-masking and <tensor> may have already been corrupted with
        # random values. If it is set to None, use <tensor>
        if truth_values is None:
            truth_values = tensor
        labels = truth_values.clone()

        probability_matrix = torch.full(labels.shape, mask_percentage)
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            labels, already_has_special_tokens=True
        )
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )
        if self.tokenizer.pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        if prefilled_labels is None:
            labels[~masked_indices] = -100  # We only compute loss on masked tokens
        else:
            labels[~masked_indices] = prefilled_labels[~masked_indices]

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        tensor[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        tensor[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return tensor, labels

    def num_chunks_for_index(self, i):
        text = self.raw_data[i]["text"]
        target = self.raw_data[i]["target"]
        with torch.no_grad():
            target_len = target.shape[0]
            if target_len > self.max_gen_len:
                target_len = self.max_gen_len
            text_len_per_chunk = (
                self.max_chunk_len
                - target_len
                - 2  # -2 for <bos> and <eos> per-target prepended.
            )
            return math.ceil(
                (text.shape[0] + 2) / text_len_per_chunk
            )  # +2 for <bos> and <eos> appended on text.

    # The output of this Dataloader is a dict as follows:
    # 'input_ids':        A list of tokenized strings (chunks) with the target string append on the end after a <CLS> token.
    # 'attention_masks':  A list of attention_masks which can be fed into the model alongside input_ids.
    # For auto-regressive language modeling (e.g. pre-training):
    # 'input_ids_masked'  Same as 'input_ids', except parts are masked randomly.
    # 'labels':           A list of either (a) masked tokens or (b) -100 for auto-regressive LM loss calculation.
    def __getitem__(self, index):
        classifier = None
        if self.includes_classification:
            classifier = self.raw_data[index]["classifier"]
        return self.process_element(
            self.raw_data[index]["text"], self.raw_data[index]["target"], classifier
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
        chk_sz = len(batch["input_ids"])
        chk_it = 0
        target_len = batch["target_len"][0][b]
        for input_ids, att_mask, masked_input_ids, perm_mask, labels, token_type_ids, target_mapping in zip(
            batch["input_ids"],
            batch["attention_masks"],
            batch["input_ids_masked"],
            batch["permutation_masks"],
            batch["labels"],
            batch["token_type_ids"],
            batch["target_mappings"],
        ):
            print("************CHUNK %i/%i***********" % (chk_it + 1, chk_sz))
            chk_it += 1
            print("target=%s" % (tokenizer.decode(input_ids[b])))
            print("token types=%s" % (token_type_ids[b].tolist()))
            print("***********************************")
            print(
                ">>>>>>>>>>PERMUTATION/ATTENTION MASK VIEW. SHOULD SEE NO PADS (only <unk>) IN TEXT, PROPER MASKING OF INDIVIDUAL TARGET ELEMENTS."
            )
            for t in range(target_len + 5):  # +5 to take a look at a few of the masks for the text tokens.
                # perm_mask is 1 where masked, 0 where not masked. we need to invert that to make the masking easy.
                perm_mask_list = perm_mask[b][t].tolist()
                perm_mask_list = [1.0 if ele == 0 else 0.0 for ele in perm_mask_list]
                perm_mask_inverted = torch.tensor(perm_mask_list, dtype=torch.float)
                masked = masked_input_ids[b] * att_mask[b] * perm_mask_inverted
                print(tokenizer.decode(masked))
            print(
                ">>>>>>>>>>TARGET VIEW. FIRST IS FULL MASKED INPUT. SECOND IS LABELS. LABELS SHOULD BE <unk> WHERE NOT MASKED, ACTUAL TOKEN WHERE <MASK>ED."
            )
            print(tokenizer.decode(masked_input_ids[b]))
            labels_fixed = labels[b].tolist()
            labels_fixed = [0.0 if lbl == -100 else lbl for lbl in labels_fixed]
            print(tokenizer.decode(labels_fixed))
            print("***********************************")


# Check a bunch of things:
# - Are there masks? Are they in a range of expected proportions?
# - If we use labels to restore input_ids_masked, do we get input_ids?
# - Does padding only occur in the last chunk and does it occur on the left? (using attention mask)
# - Does the sequence contain <bos>, <eos> and <sep>?
# These aren't conclusive tests, but they give a good sense for whether or not this algorithm is working as
# expected.
# Note that these test are not optimized at all. They run EXTREMELY slowly, like up to a minute per sequence on
# a top end PC.
def perform_tests(batch, chk_sz):
    batchsz = batch["input_ids"][0].shape[0]
    masks = 0
    target_toks = 0
    for c in range(chk_sz):
        print("chunk %i" % (c))
        for b in range(batchsz):
            if random.uniform(0, 1) < 0.9:
                # Just skip most of these. We only need a few of them for testing.
                continue
            print("batch element %i" % (b))
            ii = batch["input_ids"][c][b].tolist()
            iim = batch["input_ids_masked"][c][b].tolist()

            seq_len = len(ii)
            target_len = batch["target_len"][c][b]

            masks += iim.count(tokenizer.mask_token_id)
            target_toks += target_len

            labels = batch["labels"][c][b]
            inputs = batch["input_ids"][c][b]
            mask = batch["attention_masks"][c][b]
            perm_mask = batch["permutation_masks"][c][b]
            Check(perm_mask.shape).equals((len(ii), len(ii)))

            # Check <bos> and <eos> tokens are placed as expected on the target.
            Check(inputs[0]).equals(tokenizer.bos_token_id)
            assert tokenizer.eos_token_id in ii[1:target_len - 1]

            # Each chunk should start with <bos>
            if c == 0:
                # If padding is left-aligned, <bos> can be far into the text sequence.
                assert tokenizer.bos_token_id in ii[target_len:]
            # And should have an <eos> in its last chunk
            elif c == chk_sz - 1:
                assert tokenizer.eos_token_id in ii

            # Iterate over each token position.
            for i in range(seq_len):
                if len(labels) == len(iim):  # Cursory check for if target_mapping_labels=True
                    # Check that we can find the proper input tokens in either the labels or the masked input ids.
                    if labels[i] == -100:
                        Check(iim[i]).equals(inputs[i])
                    else:
                        Check(labels[i]).equals(inputs[i])

                # Check attention mask.
                if i > target_len:  # Target can have pads, but no attention mask to avoid giving details to the model.
                    Check(mask[i]).equals(0 if inputs[i] == tokenizer.pad_token_id else 1)

                # Check permutation mask has 0's (unmasked) for all previous target elements and 1's (masked) for
                # all following target elements, and 0's for all text elements. Text elements should mask all
                # target elements.
                for sq in range(seq_len):
                    # If we're looking at a text token, all targets should be masked.
                    if i >= target_len and sq < target_len:
                        Check(perm_mask[i][sq]).equals(1.0)
                    # If we're looking at a target token, subsequent targets should be masked.
                    elif i < target_len and sq >= i and sq < target_len:
                        Check(perm_mask[i][sq]).equals(1.0)
                    else:
                        Check(perm_mask[i][sq]).equals(0.0)


def test_against_real_file(test_file, tokenizer):
    batchsz = 16
    dataset = ChunkedTextDataset(
        data_file=test_file,
        tokenizer=tokenizer,
        max_chunk_len=256,
        max_gen_len=32,
        mask_target_percentage=0,
        mask_all_percentage=0,
        pad_left=True,
        force_max_len_gen=True,
        target_mapping_labels=True
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
