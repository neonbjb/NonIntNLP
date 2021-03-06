from transformers import XLNetTokenizer
import glob, os
from multiprocessing import Pool
import random
import torch
import csv

csv.field_size_limit(400000000)

# This function processes news articles gathered from this Kaggle dataset:
# https://www.kaggle.com/snapcrack/all-the-news/data
#
# This dataset is a CSV with the following columns:
# Index, ID, Title, Publication, Author, Publication Date, Pub Year, Pub Month, URL, Textual Content


# Converts a single line of the CSV file into a map that contains 'title', 'content'.
# Blocks content that is less than a certain character count, since this dataset has invalid content.
def process_csv_line(line):
    TITLE_INDEX = 2
    PUBLICATION_NAME = 3
    CONTENT_INDEX = 9
    element = next(csv.reader([line]))

    # This dataset honestly kind of sucks. There are a ton of invalid lines.
    if len(element) != 10:
        return None

    text = element[CONTENT_INDEX]
    pub_name = element[PUBLICATION_NAME]
    title = element[TITLE_INDEX]

    # Don't accept content with too small of text content or title content. Often these are very bad examples.
    if len(text) < 1024:
        return None
    if len(title) < 30:
        return None

    # The publication name often appears in the title. Remove it.
    # Dataset specific hack:
    if pub_name == "New York Times":
        pub_name = "The New York Times"
    title = title.replace(pub_name, "").strip()
    # This will often leave a dash prepended or appended. fix that too.
    if title.startswith("-"):
        title = title[1:].strip()
    if title.endswith("-"):
        title = title[:-1].strip()

    return {"title": title, "content": text}


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
#    'title' { input_ids_as_tensor },
#    'false_title' { input_ids_as_tensor }, <-- A string of text of the same length as title randomly sampled from text.
#    }
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

    false_start = random.randint(0, len(text) - len(title) - 1)
    false_title = text[false_start:false_start + len(title)]
    false_title_enc = tok.encode(
        false_title, add_special_tokens=False, max_length=None, pad_to_max_length=False
    )

    # Push resultants to a simple list and return it
    return {
        "text": torch.tensor(text_enc, dtype=torch.long),
        "target": torch.tensor(title_enc, dtype=torch.long),
        "false_target": torch.tensor(false_title_enc, dtype=torch.long),
    }

def corrupt_news(tokenized_news):
    for news in tokenized_news:
        decision = random.random()
        if decision < .5:
            news["classifier"] = 1.0
        elif decision >= .5 and decision < .75:
            news["target"] = news["false_target"]
            news["classifier"] = 0.0
        else:
            news["target"] = tokenized_news[random.randint(0, len(tokenized_news)-1)]["target"]
            news["classifier"] = 0.0
        del news["false_target"]


if __name__ == "__main__":
    # Fetch the news.
    folder = "C:/Users/jbetk/Documents/data/ml/title_prediction/"
    os.chdir(folder + "all-the-news/")
    files = glob.glob("*.csv")
    output_folder = "/".join([folder, "classification_outputs"])

    # Basic workflow:
    # process_files individually and compile into a list.
    # MAP: [single list of shuffled news] => map_tokenize_news
    # REDUCE: [tokenized results] => [single list of tokenized results]
    all_texts = []
    print("Reading from files..")
    for f in files:
        all_texts.extend(process_file(f))
    print("Tokenizing news..")
    p = Pool(20)
    all_news = p.map(map_tokenize_news, all_texts)

    print("Corrupting news and adding classifier..")
    corrupt_news(all_news)

    print("Writing news to output file.")
    random.shuffle(all_news)
    val_news = all_news[0:2048]
    test_news = all_news[2048:6144]
    train_news = all_news[6144:]

    torch.save(train_news, "/".join([output_folder, "train.pt"]))
    torch.save(val_news, "/".join([output_folder, "val.pt"]))
    torch.save(test_news, "/".join([output_folder, "test.pt"]))
