import transformers
import torch
import math

model_name = "xlnet-base-cased"

CHUNK_SEQ_LEN = 256
TITLE_PRED_MAX_LEN = 32

# Load model
output_dir = "C:/Users/jbetk/Documents/data/ml/saved_models/xlnet_title_generation/colab-title-first"
tokenizer = transformers.XLNetTokenizer.from_pretrained(model_name)
config = transformers.XLNetConfig.from_pretrained(model_name)
config.mem_len = 1024
model = transformers.XLNetLMHeadModel.from_pretrained(output_dir, config=config)
model.eval()

# Load some test data
test_data_file = "C:/Users/jbetk/Documents/data/ml/title_prediction/all-the-news/articles1.csv"

# Create inputs to the model for a given chunk of input_ids and a partially generated output.
def create_inputs_for_chunk(_chunk: torch.Tensor,
                            _outputs_so_far: torch.Tensor,
                            _mems: torch.Tensor,
                            _tokenizer: transformers.PreTrainedTokenizer,
                            _test_device: torch.device):
    _input_ids = torch.cat([_outputs_so_far, _chunk]).unsqueeze(dim=0)
    _inputs = {
        "input_ids": _input_ids.to(_test_device)
    }
    if _mems is not None:
        _inputs["mems"] = _mems
    return _inputs


# Returns top-k words the model predicts given _text_tensor and computed _outputs_so_far.
def predict_words(_text_tensor: torch.Tensor,
                  _outputs_so_far: torch.Tensor,
                  _predict_index: int,
                  _tokenizer: transformers.PreTrainedTokenizer,
                  _test_model: transformers.PreTrainedModel,
                  _seq_len: int,
                  _title_len: int,
                  _test_device: torch.device,
                  _k_count=3):
    _max_text_len = _seq_len - _title_len
    _chunk_count = math.ceil(_text_tensor.shape[0] / _max_text_len)
    _seq_len_needed = _chunk_count * _max_text_len
    _pads_needed = _seq_len_needed - _text_tensor.shape[0]
    _pad_tensor = torch.full((_pads_needed,), _tokenizer.pad_token_id, dtype=torch.long)
    _text_tensor = torch.cat([_pad_tensor, _text_tensor])

    _tok_text_chunked = torch.chunk(_text_tensor, _chunk_count, dim=0)
    _mems = None
    for _chunk in reversed(_tok_text_chunked):
        _inputs = create_inputs_for_chunk(_chunk, _outputs_so_far, _mems, _tokenizer, _test_device)
        _logits, _mems = _test_model.forward(**_inputs)

    _p_sft = torch.softmax(_logits[0], dim=-1)
    _probs, _words = torch.topk(_p_sft, 1)
    print("Last chunk with completion: '%s'" % _tokenizer.decode(_words))

    # Remove the batch dimension.
    _logits = torch.softmax(_logits[0][_predict_index], dim=-1)
    return torch.topk(_logits, _k_count)


def predict_forward(_text_tensor: torch.Tensor,
                    _predict_tensor: torch.Tensor,
                    _predict_index: int,
                    _tokenizer: transformers.PreTrainedTokenizer,
                    _test_model: transformers.PreTrainedModel,
                    _seq_len: int,
                    _title_len: int,
                    _test_device: torch.device):
    if _predict_index == _title_len:
        return _predict_tensor
    _probs, _words = predict_words(_text_tensor, _predict_tensor, _predict_index, _tokenizer, _test_model, _seq_len, _title_len, _test_device,
                           3)
    for _prob, _word in zip(_probs, _words):
        print("Predict %s at %f probability" % (_tokenizer.decode([_word]), _prob))
    _predict_tensor[_predict_index] = _words[0]
    if _predict_tensor[_predict_index] == _tokenizer.sep_token_id:
        return _predict_tensor
    return predict_forward(_text_tensor, _predict_tensor, _predict_index + 1, _tokenizer, _test_model, _seq_len,
                           _title_len, _test_device)


def test_model(_text_input: str,
               _tokenizer: transformers.PreTrainedTokenizer,
               _test_model: transformers.PreTrainedModel,
               _seq_len: int,
               _title_len: int,
               _test_device: torch.device):
    with torch.no_grad():
        _tok_text = torch.tensor(
            _tokenizer.encode(_text_input + _tokenizer.eos_token, add_special_tokens=False),
            dtype=torch.long)
        _tok_title = torch.full((_title_len,), _tokenizer.mask_token_id, dtype=torch.long)
        _tok_title[0] = tokenizer.bos_token_id
        _predicted_tensor = predict_forward(_tok_text, _tok_title, 1, _tokenizer, _test_model, _seq_len, _title_len,
                                            _test_device)
        return _tokenizer.decode(_predicted_tensor)

def process_csv_line(line):
    TITLE_INDEX = 2
    CONTENT_INDEX = 9
    splitted = line.split(",")

    # Once the "content" line begins, any number of commas can appear before the newline which must not be parsed.
    rejoined_content = ",".join(splitted[CONTENT_INDEX:-1])

    # Don't accept content with too small of text content or title content. Often these are very bad examples.
    if len(rejoined_content) < 1024:
        return None
    if len(splitted[TITLE_INDEX]) < 30:
        return None

    return {"title": splitted[TITLE_INDEX], "content": rejoined_content}

#for i in range(10):
#    print(test_model(article_text4, tokenizer, model, CHUNK_SEQ_LEN, TITLE_PRED_MAX_LEN, torch.device("cuda")))

model.to("cuda")
with open(test_data_file, encoding="utf-8") as file:
    line = file.readline()
    while line:
        processed = process_csv_line(line)
        if processed is not None:
            print("Input text: " + processed["content"])
            print(test_model(processed['content'], tokenizer, model, CHUNK_SEQ_LEN, TITLE_PRED_MAX_LEN, torch.device("cuda")))
            print("Actual title: " + processed['title'])
            print("\n\n")
        line = file.readline()