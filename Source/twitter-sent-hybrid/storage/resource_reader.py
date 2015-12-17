import numpy as np
import utils.utils as u


bing_liu_negative_path = "../data/lexica/BingLiu/negative-words.txt"
bing_liu_positive_path = "../data/lexica/BingLiu/positive-words.txt"
afinn_lexicon_path = "../data/lexica/AFINN/AFINN-111.txt"
mpqa_lexicon_path = "../data/lexica/MPQA/subjclueslen1-HLTEMNLP05.tff"
nrc_emoticon_path = "../data/lexica/NRC-Emotion-Lexicon-v0.92/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt"
brown_word_cluster_path = "../data/dictionaries/50mpaths2.txt"
twitter_negation_corpus_path = "../data/negation_corpus/twitter_negation_corpus.xml"


def get_automated_lexicon(filename):
    lexicon = {}
    with open(filename, 'r') as f:
        for line in f.read().decode('utf-8').split("\n"):
            newLine = line.split("\t")
            lexicon[newLine[0]] = float(newLine[1])
    return lexicon


def get_afinn_lexicon():
    return {word: int(score)
            for word, score in [line.split("\t") for line in open(afinn_lexicon_path, 'r').read().decode('utf-8').split("\n")]}

def get_bing_liu_lexicon():
    lexicon = {str(word): -3 for word in open(bing_liu_negative_path).read().decode('utf-8').split("\n")}
    lexicon.update({str(word): 3 for word in open(bing_liu_positive_path).read().decode('utf-8').split("\n")})
    return lexicon

def get_mpqa_lexicon():
    lexicon = {}
    for newLine in [line.split(" ") for line in open(mpqa_lexicon_path, 'r').read().decode('utf-8').split("\n")]:
        if newLine[5].split("=", 1)[1] == 'positive':
            lexicon[newLine[2].split("=", 1)[1]] = 2 if newLine[0][5:] == 'strongsubj' else 1
        elif newLine[5].split("=", 1)[1] == 'negative':
            lexicon[newLine[2].split("=", 1)[1]] = -2 if newLine[0][5:] == 'strongsubj'else -1
    return lexicon

def get_nrc_emotion_lexicon():
    lexicon = {}
    for newLine in [line.split("\t") for line in open(nrc_emoticon_path, 'r').read().decode('utf-8').split("\n")]:
        if newLine[2] == "1":
            if newLine[1] == 'positive':
                lexicon[newLine[0]] = 1
            elif newLine[1] == 'negative':
                lexicon[newLine[0]] = -1
    return lexicon


def get_brown_cluster_dict():
    return dict(line.split("\t")[1::-1] for line in open(brown_word_cluster_path, 'r').read().decode('utf-8').split("\n"))


def get_data(train_set, test_set):
    train = read_tsv(train_set)
    test = read_tsv(test_set)

    #train = u.reduce_dataset(train, 3000)
    return train, test

def read_tsv(filename):
    data = np.array([line.split("\t") for line in open(filename).read().decode("ISO8859-16").split("\n") if len(line) > 0])
    return u.normalize_test_set_classification_scheme(data)
