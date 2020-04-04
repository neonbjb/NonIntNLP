import glob, os
from multiprocessing import Pool
import random
import torch
import orjson
import transformers
from functools import reduce

# This function processes wikipedia articles which were pre-processed from a dump using this tool:
# https://www.kdnuggets.com/2017/11/building-wikipedia-text-corpus-nlp.html
#
# There are several articles per-file inside of a nested directory structure. Each file contains
# several JSON objects, each of which represents an article. Here is the JSON object structure:
# { "id", "url", "title", "text" }
#
# This dataset needs the following cleaning:
# 1) The article title is often repeated at the start of "text". Remove this.
# 2) The text often ends with something that looks like this:
#      "<templatestyles src=\"Refbegin/styles.css\" />\n<templatestyles src=\"Refbegin/styles.css\" />\n\n"
#    Repeated several times. These should all be removed.
#
# This processor throws out most of the metadata and splits the dataset into individual sentences. These raw sentences
# are then outputted in order. Stage 2 processors can pick them up and append them together as necessary.
#
# Note that this pre-processor has some shortcomings. Most notably (as far as I'm aware) - it splits "sentences"
# along periods. This means that abbreviations generally get separated as sentences (bad), as well as list
# bullet points (less bad). It is assumed that the downstream data processor re-combines these sentences into as
# long a length as possible, which should ameliorate these concerns somewhat.
#
# DO NOT RUN ON WINDOWS. I DONT KNOW WHY.

JUNK_TEXT = ["<templatestyles src=\"Refbegin/styles.css\" />"]

tok = transformers.AlbertTokenizer.from_pretrained("albert-large-v2")

# Processes the contents of a Wiki file and returns a list of tensors for each sentence in that text.
def map_read_files(filepath):
    with open(filepath, mode="r", encoding="utf-8") as file:
        results = []
        for line in file:
            article = orjson.loads(line)
            title = article["title"]
            text = article["text"]

            if text.startswith(title):
                text = text[len(title):]

            for junk in JUNK_TEXT:
                text = text.replace(junk, "")
            text = text.strip()

            # Also split the text into sentences.
            sentences = text.split(".")
            for sentence in sentences:
                encoded_sent = tok.encode(sentence.strip() + ".", max_length=None, pad_to_max_length=False, add_special_tokens=False)
                if len(encoded_sent) < 10 or len(encoded_sent) > 256:
                    continue
                results.append(encoded_sent)
    return results


if __name__ == "__main__":
    # Fetch the news.
    folder = "C:\\Users\\jbetk\\Documents\\data\\ml\\wiki\\out"
    output_folder = "E:\\data\\wiki\\processed"

    # Grab all the files.
    files = []
    for root, dirs, subfiles in os.walk(folder):
        files.extend([os.path.join(root, f) for f in subfiles])

    with Pool(12) as p:
        # Basic workflow:
        # MAP: [list of files to process] => list_of_news
        print("Reading from files..")
        lists_of_lists = p.map(map_read_files, files)
    all_texts = reduce(lambda r, l: r+l, lists_of_lists)

    print("Recombining lists..")
    all_texts = []
    for list in lists_of_lists:
        if list is not None:
            all_texts.extend(list)
    del lists_of_lists
    lists_of_lists = None

    print("Writing news to output file.")
    val_articles = all_texts[0:10000]
    test_articles = all_texts[10000:30000]
    train_articles = all_texts[30000:]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    torch.save(train_articles, "/".join([output_folder, "train.pt"]))
    torch.save(val_articles, "/".join([output_folder, "val.pt"]))
    torch.save(test_articles, "/".join([output_folder, "test.pt"]))
