import torch
import numpy
import transformers


# This class contains all of the logic for corrupting a "good" encoded text sequence in such a way that we can train
# a language model from just coherent text with no labeling.
class LanguageModelerScheme:
    # Initializes the class with a scheme.
    # scheme_name=should be the name of one of the methods in this class. This is the method that will be used to
    #             corrupt input text.
    # tokenizer=tokenizer to use in corruption.
    # scheme_params=most corruption schemes have configurable params. This is a dict passed in with those params. See
    #               the individual scheme methods to know what param options you have.
    def __init__(self, scheme_name, tokenizer=transformers.PreTrainedTokenizer, scheme_params=dict):
        self.lm_function = getattr(self, scheme_name)
        self.tokenizer = tokenizer
        if self.lm_function is None:
            raise EnvironmentError("Could not find specified language modeling scheme %s" % (scheme_name,))
        self.scheme_params = scheme_params

    # Unified method that can be called to generate corrupted input_ids and labels, regardless of the scheme selected.
    def corrupt_and_label(self, tokenized_string=torch.tensor):
        return self.lm_function(tokenized_string, **self.scheme_params)

    # Modeling scheme adapted from the T5 paper: https://arxiv.org/pdf/1910.10683.pdf
    # Corrupts random spans of text with mean `mean_span_len` and standard deviation `stddev_span_len`. Those spans
    # are replaced with a sentinel token. Labels consist of a string with the sentinels followed by the span that
    # was removed by that sentinel.
    def t5(self, input_ids=torch.tensor, corruption_percentage=.15, mean_span_len=3, stddev_span_len=1.5):
        seq_len = input_ids.shape[-1]
        tokens_to_remove = seq_len * corruption_percentage
        spans_to_remove = int(tokens_to_remove / mean_span_len)
        spans_to_remove = max(spans_to_remove, 1)

        indices = numpy.sort(numpy.random.randint(0, seq_len, (spans_to_remove,)))
        label_list = []
        returned_inputs = []
        last_index = 0
        for i in range(spans_to_remove):
            # If we already covered this token in a span, don't cover it more.
            if last_index != 0 and indices[i] <= last_index:
                continue

            sentinel_token_id = self.tokenizer.encode("<extra_id_%i>" % (i+1))[0]
            span_len = max(int(numpy.random.normal(mean_span_len, stddev_span_len)), 1)
            span_end = min(indices[i] + span_len, seq_len)

            label_list.append(torch.tensor([sentinel_token_id]))
            label_list.append(input_ids[indices[i]:span_end])

            if indices[i] != 0:
                returned_inputs.append(input_ids[last_index:indices[i]])
            returned_inputs.append(torch.tensor([sentinel_token_id]))
            last_index = span_end

        returned_inputs.append(input_ids[last_index:])
        return torch.cat(returned_inputs, dim=-1), torch.cat(label_list, dim=-1)


# Basic test that generates a corrupted input_id and label and prints it out, then also verifies that the bincount
# of each token before and after corruption hasn't changed. This isn't foolproof, but across a sufficiently diverse
# input and number of samples, it should be pretty conclusive.
def test_scheme(sentence, tokenizer, scheme, params={}):
    numpy.random.seed(12345)
    ids = torch.tensor(tokenizer.encode(sentence))
    orig_bincount = numpy.bincount(ids.numpy())
    l = LanguageModelerScheme(scheme, tokenizer, params)
    for i in range(5000):
        corrupted_ids, lbls = l.corrupt_and_label(ids)
        print(tokenizer.decode(corrupted_ids))
        print(tokenizer.decode(lbls))
        print("\n")
        new_bincount = numpy.bincount(corrupted_ids.numpy()) + numpy.bincount(lbls.numpy())
        new_bincount = new_bincount[:orig_bincount.shape[0]]
        if not numpy.all(numpy.equal(new_bincount, orig_bincount)):
            raise EnvironmentError("Bincount mismatch between two arrays indicates that there was a masking error.")


if __name__ == "__main__":
    sentence = """
    T5 is an encoder-decoder model pre-trained on a multi-task mixture of unsupervised and supervised tasks and for 
    which each task is converted into a text-to-text format. T5 works well on a variety of tasks out-of-the-box by 
    prepending a different prefix to the input corresponding to each task, e.g.: for translation: translate English 
    to German: or summarize:  For more information about which prefix to use, it is easiest to look into Appendix D 
    of the paper ."""
    test_scheme(sentence, transformers.AutoTokenizer.from_pretrained("t5-base"), "t5")
