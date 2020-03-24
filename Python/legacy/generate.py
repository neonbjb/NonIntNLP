import transformers
import torch
import math
import json
import os
import tqdm

class ChunkedGenerator:
    def __init__(self, max_seq_len, max_gen_len, tokenizer, model, repetition_penalty, device, branch_threshold=.5, use_token_types=False):
        self.max_seq_len = max_seq_len
        self.max_gen_len = max_gen_len
        self.tokenizer = tokenizer
        self.model = model
        self.repetition_penalty = repetition_penalty
        self.device = device
        self.k_count = 3
        self.branch_threshold = branch_threshold  # Min probability (as a percent of the top probability) assigned to a top-k prediction to justify branching.
        self.max_predictions = 24  # Maximum number of predictions this generator will make. After reaching this point, it will cease branching.
        self.use_token_types = use_token_types

    # Create inputs to the model for a given chunk of input_ids and a partially generated output.
    def create_inputs_for_chunk(
        self,
        _chunk: torch.Tensor,
        _outputs_so_far: torch.Tensor,
        _current_prediction_index: int,
        _mems: torch.Tensor,
    ):
        _batch_count, _output_len = _outputs_so_far.shape
        # Chunk has no batch tensor, so add it and expand into it.
        _chunk_expanded = _chunk.unsqueeze(0)
        _chunk_expanded = _chunk_expanded.expand(_batch_count, _chunk_expanded.shape[-1])
        _input_ids = torch.cat([_outputs_so_far, _chunk_expanded], dim=-1)

        _target_map = torch.zeros((1, 1, self.max_seq_len), dtype=torch.float)
        _target_map[0][0][_current_prediction_index] = 1.0
        _target_map = _target_map.expand((_batch_count, 1, self.max_seq_len))

        _perm_mask_text = torch.zeros((1, self.max_seq_len, _chunk.shape[-1], ), dtype=torch.float)
        _perm_mask_target = torch.ones((1, self.max_seq_len, _output_len), dtype=torch.float)
        for t_index in range(_output_len):
            _perm_mask_target[0][t_index][0:t_index] = 0.0
        _perm_mask = torch.cat([_perm_mask_target, _perm_mask_text], dim=-1).expand((_batch_count, self.max_seq_len, self.max_seq_len))

        _inputs = {
            "input_ids": _input_ids.to(self.device),
            "perm_mask": _perm_mask.to(self.device),
            "target_mapping": _target_map.to(self.device)
        }

        if self.use_token_types:
            _token_types = torch.cat(
                [
                    torch.zeros(_outputs_so_far.shape, dtype=torch.long),
                    torch.ones(_chunk_expanded.shape, dtype=torch.long),
                ], dim=-1
            )
            _inputs["token_type_ids"] = _token_types.to(self.device)

        if _mems is not None:
            _inputs["mems"] = _mems
        return _inputs


    # Returns top-k words the model predicts given _text_tensor and computed _outputs_so_far.
    def predict_words(
        self,
        _text_tensor: torch.Tensor,
        _outputs_so_far: torch.Tensor,
        _predict_index: int
    ):
        _text_tensor_len, = _text_tensor.shape
        _batch_count, _target_tensor_len = _outputs_so_far.shape
        _max_text_len = self.max_seq_len - self.max_gen_len
        _chunk_count = math.ceil(_text_tensor_len / _max_text_len)
        _seq_len_needed = _chunk_count * _max_text_len
        _pads_needed = _seq_len_needed - _text_tensor_len
        _pad_tensor = torch.full((_pads_needed,), self.tokenizer.pad_token_id, dtype=torch.long)
        _text_tensor = torch.cat([_pad_tensor, _text_tensor])

        _tok_text_chunked = torch.chunk(_text_tensor, _chunk_count, dim=-1)
        _mems = None
        for _chunk in _tok_text_chunked:
            _inputs = self.create_inputs_for_chunk(
                _chunk, _outputs_so_far, _predict_index, _mems
            )
            _logits, _mems = self.model.forward(**_inputs)

        # Apply repetition penalty against the raw logits. This is a bit different from other algorithms: it only applies \
        # itself to the most recent 15 predictions.
        if self.repetition_penalty != 1.0:
            _scan_start = _predict_index - 15
            if _scan_start < 0:
                _scan_start = 0
            for b in range(_batch_count):
                for i in range(_scan_start, _predict_index):
                    if _logits[b][0][_outputs_so_far[b][i]] < 0:
                        _logits[b][0][_outputs_so_far[b][i]] *= self.repetition_penalty
                    else:
                        _logits[b][0][_outputs_so_far[b][i]] /= self.repetition_penalty

        _p_sft = torch.softmax(_logits, dim=-1)
        _top_k = torch.topk(_p_sft, self.k_count)

        return _top_k


    def predict_forward(
        self,
        _text_tensor: torch.Tensor,
        _predict_tensor: torch.Tensor
    ):
        for i in tqdm.tqdm(range(1,self.max_gen_len)):
            _probs, _words = self.predict_words(
                _text_tensor,
                _predict_tensor,
                i
            )
            # This is where we potentially split up _predict_tensor into different predictions based on the probabilities given.
            #
            # We will accumulate predictions to be sent into the next pass into this list, which will be catted together
            # at the end of this loop.
            _predict_list = []
            _batches = _predict_tensor.shape[0]
            for b in range(_batches):
                _individual_prediction = _predict_tensor[b].clone().detach().unsqueeze(0)

                # Always take the most likely word.
                _individual_prediction[0][i] = _words[b][0][0]
                _top_probability = _probs[b][0][0]
                _predict_list.append(_individual_prediction)

                # Only take the rest of the words if it won't cause us to go over the prediction cap.
                if (len(_predict_list) + (_batches - b) - 1) >= self.max_predictions:
                    continue

                for k in range(1, self.k_count):
                    if _probs[b][0][k] / _top_probability < self.branch_threshold:
                        # Predictions are sorted by probability. If this one doesn't make the cut, none of the following
                        # ones will.
                        break
                    _branched_prediction = _individual_prediction.clone().detach()
                    _branched_prediction[0][i] = _words[b][0][k]
                    _predict_list.append(_branched_prediction)

            _predict_tensor = torch.cat(_predict_list)
        print("Done.")
        return _predict_tensor

    def test_model(
        self,
        _text_input: str
    ):
        with torch.no_grad():
            _tok_text = torch.tensor(
                self.tokenizer.encode(
                    _text_input + self.tokenizer.eos_token, add_special_tokens=False
                ),
                dtype=torch.long,
            )
            _tok_title = torch.full(
                (self.max_gen_len,), self.tokenizer.mask_token_id, dtype=torch.long
            )
            _tok_title[0] = tokenizer.bos_token_id
            # Add a batch dimension onto the predicted text. It will get expanded as the predictions branch.
            _tok_title = _tok_title.unsqueeze(0)
            _predicted_tensors = self.predict_forward(
                _tok_text,
                _tok_title
            )
            _predictions = []
            for b in range(_predicted_tensors.shape[0]):
                _predictions.append(self.tokenizer.decode(_predicted_tensors[b]))
            return _predictions


def process_csv_line(line):
    TITLE_INDEX = 2
    CONTENT_INDEX = 9
    splitted = line.split(",")

    # Once the "content" line begins, any number of commas can appear before the newline which must not be parsed.
    rejoined_content = ",".join(splitted[CONTENT_INDEX:-1])

    # Don't accept content with too small of text content or title content. Often these are very bad examples.
    if len(rejoined_content) < 1024 or len(rejoined_content) > 5500:
        return None
    if len(splitted[TITLE_INDEX]) < 30:
        return None

    return {"title": splitted[TITLE_INDEX], "content": rejoined_content}



DEVICE = "cuda"

# Load model
output_dir = "C:/Users/jbetk/Documents/data/ml/saved_models/xlnet_trainer_checkpoints/colab_pred_perm_order_go1"
#output_dir = "C:/Users/jbetk/Documents/data/ml/saved_models/xlnet_xsum/colab_pred_perm_order_go1"

with open(os.path.join(output_dir, "chunk_config.json"), "r") as chunk_cfg:
    chunk_config = json.load(chunk_cfg)
model_name = chunk_config["model_name"]
chunk_seq_len = chunk_config["max_seq_len"]
target_pred_max_len = chunk_config["predict_len"]
use_token_types = False
if "use_token_types" in chunk_config.keys():
    use_token_types = chunk_config["use_token_types"]

tokenizer = transformers.XLNetTokenizer.from_pretrained(model_name)
config = transformers.XLNetConfig.from_pretrained(model_name)
config.mem_len = chunk_config["mem_len"]
model = transformers.XLNetLMHeadModel.from_pretrained(output_dir, config=config)
model.eval()

# Load some test data
test_data_file = (
    "C:/Users/jbetk/Documents/data/ml/title_prediction/all-the-news/articles1.csv"
)

# for i in range(10):
#    print(test_model(article_text4, tokenizer, model, CHUNK_SEQ_LEN, TITLE_PRED_MAX_LEN, torch.device(DEVICE)))

model.to(DEVICE)
generator = ChunkedGenerator(chunk_seq_len, target_pred_max_len, tokenizer, model, 10, torch.device(DEVICE), branch_threshold=.5, use_token_types=use_token_types)
with open(test_data_file, encoding="utf-8") as file:
    line = file.readline()
    meta = {
        "wandb": "https://app.wandb.ai/neonbjb/nonint-transformers-torch/runs/0jw3yvln?workspace=user-neonbjb",
        "description": "256-seq model with target prepended in a 32-character fixed buffer. batch size 4.",
    }
    results = []
    i = 0
    while line and i < 5:
        processed = process_csv_line(line)
        if processed is not None:
            print("Input text: " + processed["content"])
            output = generator.test_model(processed["content"])
            print(output)
            print("Actual title: " + processed["title"])
            print("\n\n")

            result = {}
            result["input"] = processed["content"]
            result["target"] = processed["title"]
            result["prediction"] = output
            results.append(result)
            line = file.readline()
        else:
            line = file.readline()
            continue
        i += 1

    output_json = {"meta": meta, "results": results}
    with open(output_dir + "/gen_results.json", "w", encoding="utf-8") as result_file:
        json.dump(output_json, result_file, sort_keys=True, indent=4)
