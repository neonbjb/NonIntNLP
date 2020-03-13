import transformers
import torch
import os


model_name = "xlnet-base-cased"

CHUNK_SEQ_LEN = 256
TITLE_PRED_MAX_LEN = 32

# Load model
#output_dir = "C:/drive/Projects/ML/colab/saved_models/3_13_2020_xlnet_title_gen/xlnet_title_generation"
output_dir = "C:/Users/jbetk/Documents/data/ml/saved_models/xlnet_title_generation/colab-eos-gen"
tokenizer = transformers.XLNetTokenizer.from_pretrained(model_name)
config = transformers.XLNetConfig.from_pretrained(model_name)
config.mem_len = 1024
model = transformers.XLNetLMHeadModel.from_pretrained(output_dir, config=config)
model.eval()

# Test the model.
actual_article_title = "Italy announces lockdown as global coronavirus cases surpass 105,000"
article_text = """
Italian Prime Minister Giuseppe Conte signed a decree early Sunday that will put millions of people across northern Italy under lockdown due to the novel coronavirus.
The sweeping move puts the entire Lombardy region, as well as 14 other provinces, under travel restrictions, and is one of the toughest responses implemented outside of mainland China to get the Covid-19 epidemic under control.
CNN is verifying exactly when the lockdown will go into effect.
The announcement came after Italy saw a dramatic spike of 1,247 confirmed novel coronavirus cases on Saturday, the Civil Protection Department said in a statement.
The country has now recorded 5,883 cases and 233 deaths, the most fatalities outside mainland China and the biggest outbreak in Europe.
Announcing the new measures, Conte said: "There will be an obligation to avoid any movement of people who are either entering or leaving" the affected areas. "Even within the areas moving around will occur only for essential work or health reasons," he said, according to Reuters.
While the lockdown only applies to northern Italy, other measures will be applied to the entire country. These include the suspension of schools, university classes, theaters and cinemas, as well as bars, nightclubs, and sports events. Religious ceremonies, including funerals, will also be suspended.
Other countries in Europe are also struggling to contain outbreaks as cases continue to rise.
On Saturday, France's general director of health, Jerome Salomon, confirmed 16 dead and 949 infected nationwide, and Germany now has 795 cases. The United Kingdom confirmed a second death from the novel coronavirus on Saturday, while 206 people have tested positive, British health officials said in a statement.
The World Health Organization (WHO) has called on "all countries to continue efforts that have been effective in limiting the number of cases and slowing the spread of the virus."
In a statement, the WHO said: "Allowing uncontrolled spread should not be a choice of any government, as it will harm not only the citizens of that country but affect other countries as well."
Meanwhile in China, search and rescue efforts continued on Sunday for survivors from the collapse of a hotel that was being used as a coronavirus quarantine center.
The hotel, in the southeastern city of Quanzhou, in Fujian province, came down Saturday night with 80 people inside. Four people died, one person remains in critical condition and four others are seriously injured, according to China's Ministry of Emergency Management.
"We are using life detection instruments to monitor signs of life and professional breaking-in tools to make forcible entries. We are trying our utmost to save trapped people," said Guo Yutuan, squadron leader of the Quanzhou armed police detachment's mobile unit.
The building's owner is in police custody, according to state news agency Xinhua and an investigation is underway.
"""

article_text = """
Safety concerns accompanied by strong winds and power outages in ski country forced Eldora Mountain to stay closed on Thursday, March 12. As of 7:56 AM, the resort announced the closure of all lifts due to “power outages, high winds, and safety concerns.”
Wind gusts are forecast to reach up to 20 mph in Nederland, with winds of up to 11 mph expected for the Boulder area.Elsewhere in ski country, specifically at Aspen, concerns have been raised about a possible COVID-19 outbreak. Polis has urged those older than 60 to avoid high country travel due to the virus. 
There has been no indication that COVID-19 had anything to do with the “safety concerns” that shut down Eldora.
Safety concerns that shut down Eldora are likely wind-related. While wind-related lift closures are common for resorts in the high country, full closures are typically rare. It’s reasonable to believe that the resort will reopen tomorrow if winds subside, but find official updates here.
"""

article_text = """
A little under a year and a half after it released its first pair of true wireless earbuds, Sennheiser is back with a follow-up: the Sennheiser Momentum True Wireless 2. The big improvements to the true wireless earbuds are that they support noise cancellation and have much better battery life. There are also some more minor improvements, like the fact that these earbuds are 2mm smaller than their predecessors.

The improvements in battery life are, on paper, at least, pretty impressive. You’ll now get up to seven hours of playback from the earbuds themselves (up from four hours last time around), and using the case gets you 28 hours in total (up from 12). Sennheiser also claims to have fixed the battery drain problems that some users reported with the first-generation earbuds. 
It says it’s switch to a new Bluetooth chip that “counteracts any power drain possibilities.

Rounding out the specs, the Sennheiser Momentum True Wireless 2 come with support for the Bluetooth 5.1, AAC, and AptX standards, an IPX4 water resistance rating, and have an audio passthrough mode alongside their noise-canceling mode. You can listen to just one earbud at a time, but the functionality only works with the right earbud, unfortunately. They’ll retail for $299.99 (£279.99 / €299.99), the same as their predecessors.

We were impressed with the first-generation Sennheiser Momentum True Wireless when we reviewed them back in 2018, noting at the time that they were the best-sounding true wireless earbuds we’d ever heard. However, functionally, they left a little to be desired, with flaky wireless connectivity and controls that were a little unintuitive at first. And those battery drain problems were never fully addressed. We’ll be checking out whether their successors have been able to overcome these problems in our full review, which will be on the way soon.

The Sennheiser Momentum True Wireless 2 will be available in the US and Europe in black starting in April, with a white variant following later.”
"""
import math


# Create inputs to the model for a given chunk of input_ids and a partially generated output.
def create_inputs_for_chunk(_chunk: torch.Tensor,
                            _outputs_so_far: torch.Tensor,
                            _mems: torch.Tensor,
                            _tokenizer: transformers.PreTrainedTokenizer,
                            _seq_len: int,
                            _title_len: int,
                            _test_device: torch.device):
    assert (_outputs_so_far.shape[0] == _title_len)

    _input_ids = torch.cat([_chunk, _outputs_so_far]).unsqueeze(dim=0)
    #_attention_mask = torch.cat([torch.zeros((_padding_needed,), dtype=torch.float),
    #                             torch.ones((_seq_len - _padding_needed), dtype=torch.float)]).unsqueeze(dim=0)
    _token_type_ids = torch.cat([torch.zeros((_seq_len - _title_len), dtype=torch.long),
                                 torch.ones((_title_len,), dtype=torch.long)]).unsqueeze(dim=0)
    _inputs = {
        "input_ids": _input_ids.to(_test_device),
        #"attention_mask": _attention_mask.to(_test_device),
        "token_type_ids": _token_type_ids.to(_test_device)
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
    for _chunk in _tok_text_chunked:
        _inputs = create_inputs_for_chunk(_chunk, _outputs_so_far, _mems, _tokenizer, _seq_len, _title_len, _test_device)
        _logits, _mems = _test_model.forward(**_inputs)

    _p_sft = torch.softmax(_logits[0], dim=-1)
    _probs, _words = torch.topk(_p_sft, 1)
    print("Last chunk with completion: '%s'" % _tokenizer.decode(_words))

    # Remove the batch dimension.
    _logits = torch.softmax(_logits[0][_predict_index + _chunk.shape[0]], dim=-1)
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
    if _predict_tensor[_predict_index] == _tokenizer.eos_token_id:
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
            _tokenizer.encode(_tokenizer.bos_token + _text_input + _tokenizer.sep_token, add_special_tokens=False),
            dtype=torch.long)
        _tok_title = torch.full((_title_len,), _tokenizer.mask_token_id, dtype=torch.long)
        _predicted_tensor = predict_forward(_tok_text, _tok_title, 0, _tokenizer, _test_model, _seq_len, _title_len,
                                            _test_device)
        return _tokenizer.decode(_predicted_tensor)


print(test_model(article_text, tokenizer, model, CHUNK_SEQ_LEN, TITLE_PRED_MAX_LEN, torch.device("cpu")))