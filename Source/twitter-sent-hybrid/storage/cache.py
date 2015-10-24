"""
    Module for caching trained classifier. 
"""
import pickle
import json
from hashlib import md5
from os import path

dir_path = "cache/"

def generate_filename(str_id):
    filename = md5(str(str_id)).hexdigest()
    return str(filename + ".pkl")


def save_pickle(str_id, obj, useHash=True):
    full_path = dir_path + (generate_filename(str_id) if useHash else str_id + ".pkl")
    output = open(full_path, 'wb')
    pickle.dump(obj, output, protocol=2)
    output.close()
    return obj


def load_pickle(str_id, useHash=True):
    full_path = dir_path + (generate_filename(str_id) if useHash else str_id + ".pkl")
    if not path.exists(full_path):
        return False

    # check if classifier is saved at ~/.sentimetn/classifier.pickle of some sort
    pkl_file = open(full_path, 'rb')
    return pickle.load(pkl_file)


def load_json(filename):
    return json.load(open(dir_path + filename + ".json", "rb"))