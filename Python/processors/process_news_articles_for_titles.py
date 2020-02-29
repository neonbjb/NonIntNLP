import orjson
from transformers import XLNetTokenizer
import glob, os
from multiprocessing import Pool
import random

# This function processes news articles gathered from this Kaggle dataset:
# https://www.kaggle.com/snapcrack/all-the-news/data
#
# This dataset is a CSV with the following columns:
# Index, ID, Title, Publication, Author, Publication Date, Pub Year, Pub Month, URL, Textual Content


# Converts a single line of the CSV file into a map that contains 'title' and 'content'.
# Blocks content that is less than a certain character count, since this dataset has invalid content.
def process_csv_line(line):
    TITLE_INDEX = 2
    CONTENT_INDEX = 9
    splitted = line.split(",")

    # Once the "content" line begins, any number of commas can appear before the newline which must not be parsed.
    rejoined_content = ','.join(splitted[CONTENT_INDEX:-1])
    if len(rejoined_content) < 128:
        return None

    return {'title': splitted[TITLE_INDEX],
            'content': rejoined_content
            }

# Processes the entire file.
def process_file(filepath):
    result = []
    with open(filepath, encoding='utf-8') as file:
        line = file.readline()
        while line:
            processed = process_csv_line(line)
            if processed is not None:
                result.append(processed)
            line = file.readline()
    return result

## Tokenizer must be global because this file uses a map-reduce function without shared inter-process memory.
MAX_SEQ_LEN = 128
tok = XLNetTokenizer.from_pretrained("xlnet-base-cased")

# This is a map function for processing reviews. It returns a dict:
#  { 'input' { 'input_ids', 'attention_mask' },    # input text
#    'output' { 'input_ids', 'attention_mask' } }  # title for the input text.
# Inputs have no size constraints but will always be a multiple of MAX_SEQ_LEN.
def map_tokenize_news(processed):
    title = tok.cls_token + tok.bos_token + processed['title'] + tok.eos_token
    title_enc = tok.encode_plus(title, add_special_tokens=True, max_length=MAX_SEQ_LEN, pad_to_max_length=True,
                                return_token_type_ids=False, return_attention_mask=True)

    text = tok.bos_token + processed['content'] + tok.eos_token
    text_enc = tok.encode_plus(text, add_special_tokens=True, max_length=None, pad_to_max_length=False,
                               return_token_type_ids=False, return_attention_mask=True)
    # Pad to a multiple of MAX_SEQ_LEN. Pad left for XLNet. (why Google...)
    insertion_index = int(len(text_enc['input_ids']) / MAX_SEQ_LEN) * MAX_SEQ_LEN
    while len(text_enc['input_ids']) % MAX_SEQ_LEN is not 0:
        text_enc['input_ids'].insert(insertion_index, tok.pad_token_id)
        text_enc['attention_mask'].insert(insertion_index, 0)

    # Push resultants to a simple list and return it
    return {'text': text_enc, 'title': title_enc}


# Reduces a list of outputs from map_tokenize_reviews into a single list by combining across the given maps.
# Shuffles everything, then attempts to reduce to sets of reviews that can be broken up into the same set of MAX_SEQ_LEN
#   sequences. E.g: [<sentences that require 4 MAX_SEQ_LEN sequences>, ...
#                    <sentences that require 3 MAX_SEQ_LEN sequences>, ...
#                    <sentences that require 2 MAX_SEQ_LEN sequences>, ... ]

def reduce_tokenized_news(all_news):
    random.shuffle(all_news)
    list_of_multiples = []
    for processed in all_news:
        index = len(processed['text']['input_ids']) / MAX_SEQ_LEN
        while len(list_of_multiples) <= index:
            list_of_multiples.append([])
        list_of_multiples[int(index)].append(processed)

    # Only accept a multiple if there is at least 128 entries in it.
    list_of_multiples = [l for l in list_of_multiples if len(l) >= 128]
    return list_of_multiples

if __name__ == '__main__':
    # Fetch the news.
    folder = "C:/Users/jbetk/Documents/data/ml/title_prediction/"
    os.chdir(folder + "all-the-news/")
    files = ['test.csv']
    #files = glob.glob("*.csv")

    # Basic workflow:
    # process_files individually and compile into a list.
    # MAP: [single list of shuffled news] => map_tokenize_news
    # REDUCE: [tokenized results] => [single list of tokenized results]
    all_texts = []
    print("Reading from files..")
    for f in files:
        all_texts.extend(process_file(f))
    print("Tokenizing news..")
    p = Pool(23)
    all_news = p.map(map_tokenize_news, all_texts)
    all_news = reduce_tokenized_news(all_news)
    print("Combining tokenized news")

    print("Writing news to output file.")
    val_news = []
    train_news = []
    # Pull 64 articles from each multiple set for validation purposes.
    for multiple_list in all_news:
        val_news.append(multiple_list[0:32])
        train_news.append(multiple_list[32:-1])

    # Push the news to an output file.
    with open(folder + "outputs/processed.json", "wb") as output_file:
        output_file.write(orjson.dumps(train_news))
        output_file.close()

    with open(folder + "outputs/validation.json", "wb") as output_file:
        output_file.write(orjson.dumps(val_news))
        output_file.close()
