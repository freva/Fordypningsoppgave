import csv
import re

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

from sklearn.preprocessing import normalize

from classifier.data import load_data
from classifier.transformers.filter_transformer import FilterTransformer
from resources import resources
from resources.neg_cacher import NegCacher
from resources.tweebo_cacher import TweeboCacher


class LexiconTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, unigrams=True, bigrams=True, norm=True):
        self.unigrams = unigrams
        self.bigrams = bigrams
        self.normalize = norm

    def _automatic_lexicon_scorer(self, raw_tweets, lexicon_name, bigram):
        # Load the lexicon from file
        lexicon_file = resources.lexica[lexicon_name]
        with open(lexicon_file, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            lexicon = {row[0]: float(row[1]) for row in reader}

        scores = np.zeros((len(raw_tweets), 4))
        for i, contexts in enumerate(raw_tweets):
            tweet_scores = []
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
            # print("{:> 7.2f} ({})".format(tweet_scores, tweet))
        return normalize(scores) if self.normalize else scores

    def _nrc_emotion(self):
        lexicon_file = resources.lexica['nrc_e']
        with open(lexicon_file, mode='r', encoding='utf-8') as f:
            lines = f.readlines()[46:]
            reader = csv.reader(lines, delimiter='\t')
        lexicon = {}
        for row in reader:
            if int(row[2]) == 1:
                if row[1] == 'positive':
                    lexicon[row[0]] = 1
                elif row[1] == 'negative':
                    lexicon[row[0]] = -1
        return lexicon

    def _bing_liu(self):
        lexicon = {}
        pos_lexicon_file = resources.lexica['bing_p']
        with open(pos_lexicon_file, mode='r', encoding='latin-1') as f:
            for word in f.readlines()[35:]:
                lexicon[word.strip()] = 1
        neg_lexicon_file = resources.lexica['bing_n']
        with open(neg_lexicon_file, mode='r', encoding='latin-1') as f:
            for word in f.readlines()[35:]:
                # Note: Some words appear as both positive and negative: envious, enviously, and enviousness.
                # They are stored as negative.
                lexicon[word.strip()] = -1
        return lexicon

    def _mpqa(self):
        lexicon = {}
        lexicon_file = resources.lexica['mpqa']
        with open(lexicon_file, mode='r', encoding='latin-1') as f:
            reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
            for row in reader:
                if row[5].split("=", 1)[1] == 'positive':
                    if row[0].split("=", 1)[1] == 'strongsubj':
                        lexicon[row[2].split("=", 1)[1]] = 2
                    else:
                        lexicon[row[2].split("=", 1)[1]] = 1
                elif row[5].split("=", 1)[1] == 'negative':
                    if row[0].split("=", 1)[1] == 'strongsubj':
                        lexicon[row[2].split("=", 1)[1]] = -2
                    else:
                        lexicon[row[2].split("=", 1)[1]] = -1
        return lexicon

    def _manual_lexicon_scorer(self, raw_tweets, lexicon_dict):
        scores = np.zeros((len(raw_tweets), 4))
        for i, contexts in enumerate(raw_tweets):
            for token in contexts:
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
        NegCacher.cache(raw_tweets)
        negated_tweets = [NegCacher.cached[raw_tweet] for raw_tweet in raw_tweets]
        automatic_lexica = [('s140_u', False),
                            ('s140_b', True),
                            ('hs_u', False),
                            ('hs_b', True)]
        matrix = self._automatic_lexicon_scorer(negated_tweets, automatic_lexica[0][0], automatic_lexica[0][1])
        for lexicon in automatic_lexica[1:]:
            matrix = np.concatenate((matrix, self._automatic_lexicon_scorer(negated_tweets, lexicon[0], lexicon[1])),
                                    axis=1)
        manual_lexica = [self._nrc_emotion,
                         self._bing_liu,
                         self._mpqa]
        for lexicon in manual_lexica:
            matrix = np.concatenate((matrix, self._manual_lexicon_scorer(negated_tweets, lexicon())), axis=1)
        return matrix


def main():
    training_file = resources.training_data
    training_data, label_data = load_data(training_file)
    training_data = FilterTransformer().fit_transform(training_data)
    TweeboCacher.cache(training_data, True)
    lex = LexiconTransformer()
    preprocessed = lex.transform(training_data[:10])
    np.set_printoptions(precision=3)
    for i, tweet in enumerate(training_data[:10]):
        print("{}: {}".format(preprocessed[i], tweet))


if __name__ == "__main__":
    main()
