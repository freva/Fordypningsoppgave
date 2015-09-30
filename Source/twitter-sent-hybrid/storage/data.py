import numpy as np

from os import path
import cache
from models import *

import utils.utils as u
import csv

"""
    Get data sets. Both train set and test set.
    Default factor for train set is 3/4. 

    E.g. on a set of 2000 entries, 1500 is used for training and 500 for testing. 
"""
train = None
test = None

def set_file_names(train_set = None, test_set = None):
    global train, test
    cache.set_training_file(train_set)

    train = readTSV(train_set)
    test = readTSV(test_set)

    return train, test


def get_full_test_set():
    global test
    return test


def get_data():
    global train, test

    test = u.normalize_test_set_classification_scheme(test)
    train = u.normalize_test_set_classification_scheme(train)

    # Normalize data?
    train = u.reduce_dataset(train, 3000)

    docs_test, y_test = test[:,3], test[:,2]
    docs_train, y_train = train[:,3], train[:,2]

    return docs_test, y_test, docs_train, y_train


def readTSV(filename):
    return np.array([line.split("\t") for line in open(filename).read().decode("windows-1252").split("\n") if len(line) > 0])

def get_cluster_dicts():
    unique_clusters = set()
    brown_dict = {}
    brown_clusters = path.realpath("../Testing/dictionaries/50mpaths2.txt")
    with open(brown_clusters, 'r') as cluster_file:
        for row in cluster_file:
            decoded =row.decode('utf-8')
            searchResult = decoded.replace('\n', "").split('\t')
            unique_clusters.add(searchResult[0])
            brown_dict[searchResult[1]] = searchResult[0]
    return dict.fromkeys(unique_clusters, 0), brown_dict