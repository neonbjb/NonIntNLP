import numpy as np

def downconvert_tf_dataset(dataset, tok, pad_token=0, max_seq_len=128):
    inputs = []
    atts = []
    toks = []
    outputs = []
    for i,m in enumerate(dataset):
        input = tok.encode_plus(m['sentence'].numpy().decode("utf-8"),\
                                      add_special_tokens=True, max_length=max_seq_len,)
        input_ids, token_type_ids = input["input_ids"], input["token_type_ids"]
        attention_mask = [0] * len(input_ids)
        
        # Pad strings to exactly max_seq_len
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        # Double-check results.
        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_seq_len, "Error with input length {} vs {}".format(
            len(attention_mask), max_seq_len
        )
        assert len(token_type_ids) == max_seq_len, "Error with input length {} vs {}".format(
            len(token_type_ids), max_seq_len
        )
        
        # Form lists.
        inputs.append(np.asarray(input_ids))
        atts.append(np.asarray(attention_mask))
        toks.append(np.asarray(token_type_ids))
        outputs.append(m['label'].numpy())
    return [np.asarray(inputs), np.asarray(atts), np.asarray(toks)], np.asarray(outputs)