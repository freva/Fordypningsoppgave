import csv
import re

import numpy as np
from resources import resources
from sklearn.base import TransformerMixin, BaseEstimator

from sklearn.preprocessing import normalize

class LexiconTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, unigrams=True, bigrams=True, norm=True, preprocessor=None):
        self.unigrams = unigrams
        self.bigrams = bigrams
        self.normalize = norm
        self.preprocessor = preprocessor


    def _automatic_lexicon_scorer(self, raw_tweets, lexicon_name, bigram):
        # Load the lexicon from file
        lexicon_file = resources.lexica[lexicon_name]
        lexicon = {}
        with open(lexicon_file, 'r') as f:
            for line in f.read().decode('utf-8').split("\n"):
                newLine = line.split("\t")
                lexicon[newLine[0]] = float(newLine[1])
        scores = np.zeros((len(raw_tweets), 4))
        for i, contexts in enumerate(raw_tweets):
            tweet_scores = []
            contexts = (self.preprocessor(contexts)).split(" ") if self.preprocessor else contexts.split(" ")
            if bigram:
                contexts = zip(contexts, contexts[1:])
            for token in contexts:
                if bigram:
                    token = ' '.join(token)
                try:
                    tweet_scores.append(lexicon[token])
                except KeyError:
                    pass
            scores[i][0] = len([score for score in tweet_scores if score != 0])
            scores[i][1] = sum(tweet_scores) if tweet_scores else 0
            scores[i][2] = max(map(abs, tweet_scores)) if tweet_scores else 0
            scores[i][3] = tweet_scores[-1] if tweet_scores else 0
        return normalize(scores) if self.normalize else scores

    def _nrc_emotion(self):
        lexicon_file = resources.lexica['nrc_e']
        lexicon = {}
        with open(lexicon_file, 'r') as f:
            for line in f.read().decode('utf-8').split("\n"):
                newLine = line.split("\t")
                if int(newLine[2]) == 1:
                    if newLine[1] == 'positive':
                        lexicon[newLine[0]] = 1
                    elif newLine[1] == 'negative':
                        lexicon[newLine[0]] = -1
        return lexicon

    def _bing_liu(self):
        lexicon = {}
        pos_lexicon_file = resources.lexica['bing_p']
        with open(pos_lexicon_file, 'r') as f:
            for line in f.read().decode('utf-8').split("\n"):
                lexicon[line.strip()] = 1
        neg_lexicon_file = resources.lexica['bing_n']
        with open(neg_lexicon_file, 'r') as f:
            for line in f.read().decode('utf-8').split("\n"):
                lexicon[line.strip()] = -1
        return lexicon

    def _mpqa(self):
        lexicon = {}
        lexicon_file = resources.lexica['mpqa']
        with open(lexicon_file, 'r') as f:
            for line in f.read().decode('utf-8').split("\n"):
                newLine = line.split(" ")
                if newLine[5].split("=", 1)[1] == 'positive':
                    if newLine[0].split("=", 1)[1] == 'strongsubj':
                        lexicon[newLine[2].split("=", 1)[1]] = 2
                    else:
                        lexicon[newLine[2].split("=", 1)[1]] = 1
                elif newLine[5].split("=", 1)[1] == 'negative':
                    if newLine[0].split("=", 1)[1] == 'strongsubj':
                        lexicon[newLine[2].split("=", 1)[1]] = -2
                    else:
                        lexicon[newLine[2].split("=", 1)[1]] = -1
        return lexicon

    def _manual_lexicon_scorer(self, raw_tweets, lexicon_dict):
        scores = np.zeros((len(raw_tweets), 4))
        for i, contexts in enumerate(raw_tweets):
            for token in contexts.split(" "):
                try:
                    negated_regex = r'(.*)_NEG(?:FIRST)?$'
                    if re.match(negated_regex, token):
                        token = re.sub(negated_regex, r'\1', token)
                        scores[i][2 if lexicon_dict[token] > 0 else 3] += lexicon_dict[token]
                    else:
                        scores[i][0 if lexicon_dict[token] > 0 else 1] += lexicon_dict[token]
                except KeyError:
                    pass
        return normalize(scores) if self.normalize else scores

    def fit(self, raw_tweets, y=None):
        return self

    def transform(self, raw_tweets, y=None):
        automatic_lexica = [('s140_u', False),
                            ('s140_b', True),
                            ('hs_u', False),
                            ('hs_b', True)]
        matrix = self._automatic_lexicon_scorer(raw_tweets, automatic_lexica[0][0], automatic_lexica[0][1])
        for lexicon in automatic_lexica[1:]:
            matrix = np.concatenate((matrix, self._automatic_lexicon_scorer(raw_tweets, lexicon[0], lexicon[1])),
                                    axis=1)
        manual_lexica = [self._nrc_emotion,
                         self._bing_liu,
                         self._mpqa]
        for lexicon in manual_lexica:
            matrix = np.concatenate((matrix, self._manual_lexicon_scorer(raw_tweets, lexicon())), axis=1)
        return matrix

