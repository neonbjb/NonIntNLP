import orjson
from transformers import (BertTokenizer, DistilBertTokenizer, GPT2Tokenizer, AlbertTokenizer)
import random
import glob, os
import time
from multiprocessing import Pool

# This function processes Amazon reviews derived from the datasets available here:
# http://jmcauley.ucsd.edu/data/amazon/
# This dataset consists of a list of new-line separated entries. Entries are maps with the following keys:
#  reviewerID
#  asin
#  reviewerName
#  helpful (list of 2 numbers)
#  reviewText
#  overall (score as a float. always [1,2,3,4,5])
#  summary
#  unixReviewTime
#  reviewTime
#
# Converts the given entry with the above format into a new map with the following keys:
#  sentence (string)
#  label (integer)
def process_amazon_entry(entry, min_helpful=0, max_words=0):
    if not 'helpful' in entry:
        return None
    if entry['helpful'][0] < min_helpful:
        return None
    if entry['reviewText'].count(' ') + 1 > max_words:
        return None
    return {'sentence': entry['reviewText'],
            'label': int(entry['overall']),
            }


# This function processes Amazon reviews derived from the reviews.json dataset available here:
# https://www.yelp.com/dataset
# This dataset consists of a list of new-line separated entries. Entries are maps with the following keys:
#  review_id, user_id, business_id b64
#  stars (float)
#  useful, funny, cool (integers)
#  text
#  date
#
# Converts the given entry with the above format into a new map with the following keys:
#  sentence (string)
#  label (integer)
def process_yelp_entry(entry, min_helpful=0, max_words=0):
    if not 'business_id' in entry:
        return None
    if entry['useful'] < min_helpful:
        return None
    if entry['text'].count(' ') + 1 > max_words:
        return None
    return {'sentence': entry['text'],
            'label': int(entry['stars']),
            }


def process_file(filepath):
    result = [[], [], [], [], []]
    with open(filepath, encoding="utf-8") as file:
        line = file.readline()
        while line:
            entry = orjson.loads(line)
            pent = None
            # Do a simple inference on the type of the file being passed in based on the fields provided.
            if 'business_id' in entry:
                pent = process_yelp_entry(entry, min_helpful=2, max_words=300)
            elif 'asin' in entry:
                pent = process_amazon_entry(entry, min_helpful=2, max_words=300)
            if pent:
                result[pent['label'] - 1].append(pent)
            line = file.readline()
    return result


# Reduce function called after reading all files.
# Input is a list of results.
# Result is a 5-entry list. Each entry corresponds with a star rating.
# Entries of result lists are further lists of actual reviews. Each review is a map including sentence and rating.
#
# This method reduces the first dimension of results by combining each 5-entry list together to produce a single
# 5-entry list. It then normalizes all 5 entries to have the same total number of entries. The entries are then
# combined together and returned.
#
# The result of this is a homogeneous list of sentences and ratings where each rating is equally represented.
def reduce_file_reads(list_of_results):
    # Actual results will be stored here.
    combined_results = [[], [], [], [], []]

    # First, reduce dimensionality of input by combining all 5-entry lists.
    for results in list_of_results:
        for (i, r) in enumerate(results):
            combined_results[i].extend(r)

    # Now, normalize the 5-entry resultant list.
    rating_counts = [0] * 5
    for (i, rl) in enumerate(combined_results):
        rating_counts[i] = len(rl)

    # take the minimum star rating, and inflate it by a factor of 1.5. Do this because mid stars tend to have a very low
    # comparative frequency. below we will cause these mid-star ratings to be repeated at random.
    min_rating_count = int(min(rating_counts) * 1.5)

    balanced_reviews = []
    for rl in combined_results:
        balanced_reviews.extend(random.choices(rl, k=min_rating_count))

    # Shuffle the reviews. We don't care about ratings from here on out.
    random.shuffle(balanced_reviews)

    print("Total reviews collected: ", len(balanced_reviews))

    return balanced_reviews

is_gpt2 = False
if is_gpt2:
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = "<|endoftext|>"
else:
    tok = AlbertTokenizer.from_pretrained("albert-large-v2")

# This is a map function for processing reviews. It returns a list of tokenized
# reviews and labels.
def map_tokenize_reviews(review):
    pad_token = 0
    max_seq_len = 128

    sentence = review['sentence']
    if is_gpt2:
        # Add a space prefix to the sentence before tokenizing for GPT-2 (and byte-encoded) models.
        sentence = " " + sentence
    input = tok.encode_plus(sentence, add_special_tokens=True, max_length=max_seq_len, pad_to_max_length=True)
    input_ids, token_type_ids = input["input_ids"], input["token_type_ids"]
    attention_mask = [0] * len(input_ids)

    # Push resultants to a simple list and return it
    return [input_ids, attention_mask, token_type_ids, review['label']]


# Reduces a list of outputs from map_tokenize_reviews into a single list by combining across the internal maps.
def reduce_tokenized_reviews(reviews):
    result = {'input_id': [],
              'attention_mask': [],
              'token_type_id': [],
              'label': [],
              }
    for review in reviews:
        for (i, k) in enumerate(result.keys()):
            result[k].append(review[i])
    return result

if __name__ == '__main__':
    # Fetch the reviews.
    folder = "C:/Users/jbetk/Documents/data/ml/sentiment_analysis/"
    os.chdir(folder + "amazon/")
    #files = []
    files = glob.glob("*.json")
    files.append(folder + "yelp/review.json")

    # Basic workflow:
    # MAP: [files] => process_file
    # REDUCE: [all reviews]->[balance reviews]->[single list of shuffled reviews]
    # MAP: [single list of shuffled reviews] => map_tokenize_reviews
    # REDUCE: [tokenized results] => [single list of tokenized results]
    p = Pool(20)
    print("Reading files & combining & normalizing..")
    all_reviews = p.map(process_file, files)
    all_reviews = reduce_file_reads(all_reviews)
    print("Tokenizing reviews..")
    all_reviews = p.map(map_tokenize_reviews, all_reviews)
    all_reviews = reduce_tokenized_reviews(all_reviews)
    print("Combining tokenized reviews")

    print("Writing reviews to output file.")
    val_reviews = {}
    train_reviews = {}
    for k in all_reviews.keys():
        val_reviews.update({k: all_reviews[k][0:4000]})
        train_reviews.update({k: all_reviews[k][4000:]})

    # Push the reviews to an output file.
    with open(folder + "outputs/processed.json", "wb") as output_file:
        output_file.write(orjson.dumps(train_reviews))
        output_file.close()

    with open(folder + "outputs/validation.json", "wb") as output_file:
        output_file.write(orjson.dumps(val_reviews))
        output_file.close()
