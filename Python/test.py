import transformers
import torch
import os
import time


model_name = "xlnet-base-cased"

CHUNK_SEQ_LEN = 128
TITLE_PRED_MAX_LEN = 64

# Load model
output_dir = os.path.join("c:/Users/jbetk/Documents/data/ml/saved_models", "xlnet_title_generation")
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
actual_article_title = "House Republicans Fret About Winning Their Health Care Suit - The New York Times"
article_text = """
WASHINGTON  —   
Congressional Republicans have a new fear when it comes to their    health care lawsuit against the Obama administration: They might win. The 
incoming Trump administration could choose to no longer defend the executive branch against the suit, which challenges the administration’s
 authority to spend billions of dollars on health insurance subsidies for   and   Americans, handing House Republicans a big victory on    issues.
  But a sudden loss of the disputed subsidies could conceivably cause the health care program to implode, leaving millions of people without access to 
  health insurance before Republicans have prepared a replacement. That could lead to chaos in the insurance market and spur a political backlash just as
   Republicans gain full control of the government. To stave off that outcome, Republicans could find themselves in the awkward position of appropriating 
   huge sums to temporarily prop up the Obama health care law, angering conservative voters who have been demanding an end to the law for years. In another 
   twist, Donald J. Trump’s administration, worried about preserving executive branch prerogatives, could choose to fight its Republican allies in the House 
   on some central questions in the dispute. Eager to avoid an ugly political pileup, Republicans on Capitol Hill and the Trump transition team are gaming 
   out how to handle the lawsuit, which, after the election, has been put in limbo until at least late February by the United States Court of Appeals for the
    District of Columbia Circuit. They are not yet ready to divulge their strategy. “Given that this pending litigation involves the Obama administration and 
    Congress, it would be inappropriate to comment,” said Phillip J. Blando, a spokesman for the Trump transition effort. “Upon taking office, the Trump
     administration will evaluate this case and all related aspects of the Affordable Care Act. ” In a potentially   decision in 2015, Judge Rosemary M. 
     Collyer ruled that House Republicans had the standing to sue the executive branch over a spending dispute and that the Obama administration had been 
     distributing the health insurance subsidies, in violation of the Constitution, without approval from Congress. The Justice Department, confident that 
     Judge Collyer’s decision would be reversed, quickly appealed, and the subsidies have remained in place during the appeal. In successfully seeking a 
     temporary halt in the proceedings after Mr. Trump won, House Republicans last month told the court that they “and the  ’s transition team currently 
     are discussing potential options for resolution of this matter, to take effect after the  ’s inauguration on Jan. 20, 2017. ” The suspension of the 
     case, House lawyers said, will “provide the   and his future administration time to consider whether to continue prosecuting or to otherwise resolve 
     this appeal. ” Republican leadership officials in the House acknowledge the possibility of “cascading effects” if the   payments, which have totaled 
     an estimated $13 billion, are suddenly stopped. Insurers that receive the subsidies in exchange for paying    costs such as deductibles and   for 
     eligible consumers could race to drop coverage since they would be losing money. Over all, the loss of the subsidies could destabilize the entire 
     program and cause a lack of confidence that leads other insurers to seek a quick exit as well. Anticipating that the Trump administration might not be 
     inclined to mount a vigorous fight against the House Republicans given the  ’s dim view of the health care law, a team of lawyers this month sought to 
     intervene in the case on behalf of two participants in the health care program. In their request, the lawyers predicted that a deal between House 
     Republicans and the new administration to dismiss or settle the case “will produce devastating consequences for the individuals who receive these 
     reductions, as well as for the nation’s health insurance and health care systems generally. ” No matter what happens, House Republicans say, they want to prevail on two overarching concepts: the congressional power of the purse, and the right of Congress to sue the executive branch if it violates the Constitution regarding that spending power. House Republicans contend that Congress never appropriated the money for the subsidies, as required by the Constitution. In the suit, which was initially championed by John A. Boehner, the House speaker at the time, and later in House committee reports, Republicans asserted that the administration, desperate for the funding, had required the Treasury Department to provide it despite widespread internal skepticism that the spending was proper. The White House said that the spending was a permanent part of the law passed in 2010, and that no annual appropriation was required  —   even though the administration initially sought one. Just as important to House Republicans, Judge Collyer found that Congress had the standing to sue the White House on this issue  —   a ruling that many legal experts said was flawed  —   and they want that precedent to be set to restore congressional leverage over the executive branch. But on spending power and standing, the Trump administration may come under pressure from advocates of presidential authority to fight the House no matter their shared views on health care, since those precedents could have broad repercussions. It is a complicated set of dynamics illustrating how a quick legal victory for the House in the Trump era might come with costs that Republicans never anticipated when they took on the Obama White House.
"""


def prepare_chunked_inputs(_batched_inputs, _max_seq_len, _max_title_len):
    # We need to do a lot more data preparation before feeding into the model.
    # First, chunk the batch into a list of tensors each of size max_seq_len.
    with torch.no_grad():
        _batch_sz = _batched_inputs[0].shape[0]
        _nr_chunks = int(_batched_inputs[0].shape[-1] / _max_seq_len)
        _batch_inputs = _batched_inputs[0:3]
        _chunked_batch_inputs_by_input_nr = [torch.chunk(_bi, _nr_chunks, -1) for _bi in _batch_inputs]
        _chunked_batch_inputs = []

        # These tensors will be used to append on to the input tensors where the prediction will occur.
        _input_mask_tensor = torch.full((_batch_sz, _max_title_len), tokenizer.mask_token_id, dtype=torch.long)
        _ones_float_tensor_for_title = torch.ones((_batch_sz, _max_title_len), dtype=torch.float)
        _ones_long_tensor_for_title = torch.ones((_batch_sz, _max_title_len), dtype=torch.long)
        for i in range(_nr_chunks):
            # For the input_ids (index 0), append on _max_title_len masks.
            _chunked_input_ids = torch.cat([_chunked_batch_inputs_by_input_nr[0][i], _input_mask_tensor], dim=-1)
            # For the attention mask, just add all 1s because this is not padding.
            _chunked_attention_mask = torch.cat([_chunked_batch_inputs_by_input_nr[1][i], _ones_float_tensor_for_title],
                                                dim=-1)
            # For token type IDs, also all 1s since this is the "second sentence".
            _chunked_token_type_ids = torch.cat([_chunked_batch_inputs_by_input_nr[2][i], _ones_long_tensor_for_title],
                                                dim=-1)
            _chunked_batch_inputs.append([_chunked_input_ids, _chunked_attention_mask, _chunked_token_type_ids])

        # Create a target mapping that will be used for all inputs, since they all follow a similar format.
        _target_mapping = torch.zeros((_batch_sz, _max_title_len, _max_seq_len + _max_title_len), dtype=torch.float)
        for i in range(_max_title_len):
            for b in range(_batch_sz):
                _target_mapping[b][i][_max_seq_len + i] = 1

        # Next, gather the expected output IDs and generate the 'labels' format that transformers is expecting.
        _labels = _batched_inputs[3]
    return _chunked_batch_inputs, _target_mapping, _labels

def chunk_to_inputs(_chunk, _target_mapping, _labels, _mems, _device):
    _inputs = {"input_ids": _chunk[0],
               "attention_mask": _chunk[1],
               "token_type_ids": _chunk[2],
               "target_mapping": _target_mapping}

    if _labels is not None:
        _inputs["labels"] = _labels

    # Don't forget to send all these tensors to the device.
    for i, (k, v) in enumerate(_inputs.items()):
        _inputs[k] = v.to(_device)

    # Mems will just stay on-device, so add them last.
    if _mems is not None:
        _inputs["mems"] = _mems
    return _inputs

def test_model(_text_input, _test_model, _test_tokenizer, _seq_len, _title_len, _test_device):
    _tokenized_text_plus = _test_tokenizer.encode_plus(_text_input, add_special_tokens=True, max_length=None,
                                                pad_to_max_length=False,
                                                return_token_type_ids=True, return_attention_mask=True)
    # Pad each input to make it a multiple of _seq_len
    insertion_index = int(len(_tokenized_text_plus['input_ids']) / _seq_len) * _seq_len
    while len(_tokenized_text_plus['input_ids']) % _seq_len is not 0:
        _tokenized_text_plus['input_ids'].insert(insertion_index, tokenizer.pad_token_id)
        _tokenized_text_plus['attention_mask'].insert(insertion_index, 0)
        _tokenized_text_plus['token_type_ids'].insert(insertion_index, 0)

    _title = _test_tokenizer.cls_token + _test_tokenizer.bos_token + actual_article_title + _test_tokenizer.eos_token
    # Insert the title as the second sentence, forcing the proper token types.
    _title_enc = _test_tokenizer.encode_plus("", _title, add_special_tokens=True, max_length=_title_len, pad_to_max_length=True,
                                return_token_type_ids=True, return_attention_mask=True)

    _test_batch = [torch.tensor(_tokenized_text_plus['input_ids'], dtype=torch.long).unsqueeze(0),
                   torch.tensor(_tokenized_text_plus['attention_mask'], dtype=torch.float).unsqueeze(0),
                   torch.tensor(_tokenized_text_plus['token_type_ids'], dtype=torch.long).unsqueeze(0),
                   torch.tensor(_title_enc['input_ids'], dtype=torch.long).unsqueeze(0)]

    _test_chunked_batch_inputs, _test_target_mapping, _test_labels = prepare_chunked_inputs(_test_batch, _seq_len,
                                                                                            _title_len)
    _test_num_chunks = len(_test_chunked_batch_inputs)

    _test_mems = None
    _test_loss = None
    _test_chunk_loss_schedule = []
    _test_logits = None
    with torch.no_grad():
        _chunk_nr = 1
        for _test_chunk in _test_chunked_batch_inputs:
            print("Processing chunk %i.." % (_chunk_nr))
            _test_inputs = chunk_to_inputs(_test_chunk, _test_target_mapping, _test_labels, _test_mems, _test_device)
            # I'm assuming no loss is returned since I'm not giving it any labels - may need to adjust this.
            _test_loss, _test_logits, _test_mems = _test_model.forward(**_test_inputs)
            _test_chunk_loss_schedule.append(_test_loss.item())
            _chunk_nr += 1
        _test_logits_argmax = torch.argmax(_test_logits, dim=-1)
        return tokenizer.decode(_test_logits_argmax[0].numpy())

print(test_model(article_text, model, tokenizer, 128, 64, "cpu"))
