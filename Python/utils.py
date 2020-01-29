from transformers import glue_convert_examples_to_features


def encode_and_shuffle_glue_dataset(dataset, tokenizer, dataset_name, max_seq_len=128, batch_size=32, shuffle_seed=1337):
    dataset = glue_convert_examples_to_features(dataset, tokenizer, max_seq_len, dataset_name)
    return dataset.shuffle(shuffle_seed).batch(batch_size).repeat