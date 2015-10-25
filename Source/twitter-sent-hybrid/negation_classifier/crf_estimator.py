from collections import namedtuple
from time import time
import pickle as pkl
import os
import pycrfsuite

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import f1_score, precision_score, recall_score
from crf_transformer import CRFTransformer

crf_params = {'c1': 0.1, 'c2': 1, 'max_distance': 7, 'max_iterations': 1000, 'linesearch': 'MoreThuente'}


class CRF(BaseEstimator, ClassifierMixin):
    def __init__(self, c1=1.0, c2=1e-3, max_iterations=1000, linesearch='MoreThuente', max_distance=5, verbose=True):
        self.c1 = c1
        self.c2 = c2
        self.max_iterations = max_iterations
        self.linesearch = linesearch
        self.max_distance = max_distance
        self.trainer = pycrfsuite.Trainer(verbose=verbose)
        self.trainer.set_params({
            'c1': self.c1,  # coefficient for L1 penalty
            'c2': self.c2,  # coefficient for L2 penalty
            'max_iterations': self.max_iterations,  # stop earlier
            'linesearch': self.linesearch,

            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })
        self.model_file = 'cache/model.crf'

    def set_params(self, **params):
        super(CRF, self).set_params(**params)
        self.trainer.set_params({
            'c1': params['c1'],
            'c2': params['c2'],
            'max_iterations': params['max_iterations'],
            'linesearch': params['linesearch'],
        })
        return self

    def fit(self, X, y):
        for xseq, yseq in zip(X, y):
            self.trainer.append(xseq, yseq)
        self.trainer.train(self.model_file)
        return self

    def predict(self, X):
        tagger = pycrfsuite.Tagger()
        tagger.open(self.model_file)
        return [tagger.tag(xseq) for xseq in X]


def sequence_score(scorer, test, predicted):
    y_test = [token for tweet in test for token in tweet]
    predicted = [token for tweet in predicted for token in tweet]
    return scorer(y_test, predicted, labels=['affirmative', 'negated'], pos_label='negated', average='binary')


def find_pcs(y_test, y_pred, tweet_cues_test, tweet_cues_pred):
    test = [[token_pair for token_pair in zip(*tweet_pair)]
            for tweet_pair in zip(y_test, tweet_cues_test)]
    predicted = [[token_pair for token_pair in zip(*tweet_pair)]
                 for tweet_pair in zip(y_pred, tweet_cues_pred)]

    conflicts = 0
    agreements = 0
    for test_tweet, pred_tweet in zip(test, predicted):
        common_scope = False
        for test_token, pred_token in zip(test_tweet, pred_tweet):
            is_cue = False
            if test_token[1] != pred_token[1]:
                conflicts += 1
            elif test_token[1] and pred_token[1]:
                common_scope = True
                is_cue = True
            if common_scope and (test_token[0] != pred_token[0]):
                conflicts += 1
                common_scope = False
            elif not is_cue and common_scope \
                    and ((test_token[0] == 'affirmative') and (pred_token[0] == 'affirmative')):
                agreements += 1
                common_scope = False
        # if tweet ended with an active common scope, mark as an agreement
        agreements += int(common_scope)
    return agreements / (conflicts + agreements)


def cross_validate(X, y, tweet_cues, num_folds, max_distance, **crf_params):
    print("Performing {}-fold cross validation on {} samples...".format(num_folds, len(X)))

    negation_tweets = ['negated' in tweet for tweet in y]
    kf = StratifiedKFold(negation_tweets, n_folds=num_folds, shuffle=True)
    precision, recall, f1, pcs = [], [], [], []

    for train, test in kf:
        t0 = time()
        X_train, y_train, tweet_cues_train = zip(*[(X[i], y[i], tweet_cues[i]) for i in train])
        X_test, y_test, tweet_cues_test = zip(*[(X[i], y[i], tweet_cues[i]) for i in test])

        transformer = CRFTransformer(max_distance)

        X_train_seq = transformer.transform(X_train, tweet_cues=tweet_cues_train)
        X_test_seq = transformer.transform(X_test)

        clf = CRF(verbose=False, **crf_params)
        clf.fit(X_train_seq, y_train)
        y_pred = clf.predict(X_test_seq)

        tweet_cues_pred = transformer.cue_detector.predict([[token[0] for token in tweet] for tweet in X_test])

        precision.append(sequence_score(precision_score, y_test, y_pred))
        recall.append(sequence_score(recall_score, y_test, y_pred))
        f1.append(sequence_score(f1_score, y_test, y_pred))
        pcs.append(find_pcs(y_test, y_pred, tweet_cues_test, tweet_cues_pred))

        print("Fold trained in {:.3f} seconds".format(time() - t0))
    scores = namedtuple('Scores', ['precision', 'recall', 'f1', 'pcs'])
    scores.precision = sum(precision) / len(precision)
    scores.recall = sum(recall) / len(recall)
    scores.f1 = sum(f1) / len(f1)
    scores.pcs = sum(pcs) / len(pcs)
    return scores

def load_data(parsed_corpus):
    X = [[(token, pos_tag, dependency) for token, pos_tag, dependency, _, _ in tweet]
         for tweet in parsed_corpus]
    tweet_cues = [[is_cue for _, _, _, is_cue, _ in tweet] for tweet in parsed_corpus]
    y = [[label for _, _, _, _, label in tweet] for tweet in parsed_corpus]
    return X, y, tweet_cues

def get_classifier(corpus, force_new=False):
    crf_path = 'cache/crf.pkl'
    if os.path.exists(crf_path) and not force_new:
        print("Loading CRF classifier from pickle...")
        with open(crf_path, 'rb') as f:
            clf = pkl.load(f)
    else:
        print("Loading and dependency parsing negation corpus...")
        X, y, tweet_cues = load_data(corpus)

        transformer = CRFTransformer(int(crf_params['max_distance']))
        y_tweet_negated = ['negated' in tweet for tweet in y]
        split = StratifiedShuffleSplit(y_tweet_negated, 1, 0.25, random_state=1)
        for train_indices, test_indices in split:
            X_train = [X[train_index] for train_index in train_indices]
            X_test = [X[test_index] for test_index in test_indices]
            y_train = [y[train_index] for train_index in train_indices]
            y_test = [y[test_index] for test_index in test_indices]
            tweet_cues_train = [tweet_cues[train_index] for train_index in train_indices]
            tweet_cues_test = [tweet_cues[test_index] for test_index in test_indices]

        X_seq_train = transformer.transform(X_train, tweet_cues=tweet_cues_train)
        X_seq_test = transformer.transform(X_test)
        print("Training new CRF classifier...")

        clf = CRF(float(crf_params['c1']), float(crf_params['c1']),
                  int(crf_params['max_iterations']), crf_params['linesearch'], verbose=False)
        print("Training CRF...")
        clf.fit(X_seq_train, y_train)

        with open(crf_path, 'wb') as f:
            pkl.dump(clf, f, protocol=2)

    return clf
