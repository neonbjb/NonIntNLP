from transformers import XLNetTokenizer
import glob, os
from multiprocessing import Pool
import random
import torch

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
    rejoined_content = ",".join(splitted[CONTENT_INDEX:-1])

    # Don't accept content with too small of text content or title content. Often these are very bad examples.
    if len(rejoined_content) < 1024:
        return None
    if len(splitted[TITLE_INDEX]) < 30:
        return None

    return {"title": splitted[TITLE_INDEX], "content": rejoined_content}


# Processes the entire file.
def process_file(filepath):
    result = []
    with open(filepath, encoding="utf-8") as file:
        line = file.readline()
        while line:
            processed = process_csv_line(line)
            if processed is not None:
                result.append(processed)
            line = file.readline()
    return result


tok = XLNetTokenizer.from_pretrained("xlnet-base-cased")

# This is a map function for processing reviews. It returns a dict:
#  { 'text' { input_ids_as_tensor },
#    'title' { input_ids_as_tensor } }
def map_tokenize_news(processed):
    text = processed["content"]
    text_enc = tok.encode(
        text, add_special_tokens=False, max_length=None, pad_to_max_length=False
    )

    title = processed["title"]
    # Insert the title as the second sentence, forcing the proper token types.
    title_enc = tok.encode(
        title, add_special_tokens=False, max_length=None, pad_to_max_length=False
    )

    # Push resultants to a simple list and return it
    return {
        "text": torch.tensor(text_enc, dtype=torch.long),
        "target": torch.tensor(title_enc, dtype=torch.long),
    }


if __name__ == "__main__":
    # Fetch the news.
    folder = "C:/Users/jbetk/Documents/data/ml/title_prediction/"
    os.chdir(folder + "all-the-news/")
    files = glob.glob("*.csv")
    output_folder = "/".join([folder, "outputs"])

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

    print("Writing news to output file.")
    random.shuffle(all_news)
    val_news = all_news[0:2048]
    test_news = all_news[2048:6144]
    train_news = all_news[6144:]

    torch.save(train_news, "/".join([output_folder, "train.pt"]))
    torch.save(val_news, "/".join([output_folder, "val.pt"]))
    torch.save(test_news, "/".join([output_folder, "test.pt"]))
