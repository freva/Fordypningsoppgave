"""
    Module for caching trained classifier. 
"""
from hashlib import md5
import pickle
from os import path

dir_path = "pickles/"

def generate_filename(str_id):
    filename = md5(str(str_id)).hexdigest()
    return str(dir_path + filename + ".pkl")


def save(str_id, obj):
    full_path = generate_filename(str_id)
    output = open(full_path, 'wb')
    pickle.dump(obj, output)
    output.close()
    return obj


def get(str_id):
    full_path = generate_filename(str_id)
    if not path.exists(full_path):
        return False

    # check if classifier is saved at ~/.sentimetn/classifier.pickle of some sort
    pkl_file = open(full_path, 'rb')
    return pickle.load(pkl_file)
