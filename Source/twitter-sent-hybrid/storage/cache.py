"""
    Module for caching trained classifier. 
"""
import pickle
from hashlib import md5
from os import path

dir_path = "pickles/"

def generate_filename(str_id):
    filename = md5(str(str_id)).hexdigest()
    return str(filename + ".pkl")


def save(str_id, obj, useHash=True):
    full_path = dir_path + (str_id + ".pkl") if useHash else generate_filename(str_id)
    output = open(full_path, 'wb')
    pickle.dump(obj, output, protocol=2)
    output.close()
    return obj


def get(str_id, useHash=True):
    full_path = dir_path + (str_id + ".pkl") if useHash else generate_filename(str_id)
    if not path.exists(full_path):
        return False

    # check if classifier is saved at ~/.sentimetn/classifier.pickle of some sort
    pkl_file = open(full_path, 'rb')
    return pickle.load(pkl_file)
