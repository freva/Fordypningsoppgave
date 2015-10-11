import numpy as np

import utils.utils as u

"""
    Get data sets. Both train set and test set.
    Default factor for train set is 3/4. 

    E.g. on a set of 2000 entries, 1500 is used for training and 500 for testing. 
"""


def get_data(train_set, test_set):
    train = read_tsv(train_set)
    test = read_tsv(test_set)

    test = u.normalize_test_set_classification_scheme(test)
    train = u.normalize_test_set_classification_scheme(train)

    test = u.generate_subjective_set(test)
    train = u.generate_subjective_set(train)

    # Normalize data?
    #train = u.reduce_dataset(train, 3000)

    return train, test


def read_tsv(filename):
    return np.array([line.split("\t") for line in open(filename).read().decode("windows-1252").split("\n") if len(line) > 0])


def get_cluster_dict():
    return dict(line.split("\t")[1::-1] for line in open("../Testing/dictionaries/50mpaths2.txt", 'r').read().decode('utf-8').split("\n"))

def get_sentiment_lexicon():
    lexicon = {}
    with open("../Testing/data/AFINN/AFINN-111.txt", 'r') as f:
        for line in f.read().decode('utf-8').split("\n"):
            newLine = line.split("\t")
            lexicon[(newLine[0])] = newLine[1]
    return lexicon

def get_neg_and_pos_word_lexicon():
    lexicon = {}
    with open("../Testing/data/BingLiu/negative-words.txt", 'r') as f:
        for word in f.read().decode('utf-8').split("\n"):
            lexicon[str(word)] = -3

    with open("../Testing/data/BingLiu/positive-words.txt", 'r') as f:
        for word in f.read().decode('utf-8').split("\n"):
            lexicon[str(word)] = 3

    return lexicon