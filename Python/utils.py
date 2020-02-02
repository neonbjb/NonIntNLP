import numpy as np
from queue import Queue
from threading import Thread

# Yield successive {n}-sized chunks from {lst}.
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def _map_reduce_thread(map_fn, input, q):
    q.put(map_fn(input))

# Performs a multi-threaded map-reduce operation.
# This is done by chunking {inputs} into at most {max_threads} separate lists, then
# feeding each list into a separate {map_fn} which runs on its own thread.
# Waits for each thread to return its results. Results are compiled into a list of results and
# fed to the {reduce_fn}. The result of this call is returned.
def perform_map_reduce(map_fn, reduce_fn, inputs, max_threads):
    threads = []
    thread_count = min(max_threads, len(inputs))
    chunked_inputs = chunks(inputs, int(len(inputs) / thread_count))
    q = Queue()
    for c in chunked_inputs:
        t = Thread(target=lambda fn, i, qu: qu.put(fn(i)), args=(map_fn, c, q))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    return reduce_fn(list(q.queue))


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