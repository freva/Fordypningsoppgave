import re
from sklearn.base import BaseEstimator, ClassifierMixin
from utils import filters


class CueDetector(BaseEstimator, ClassifierMixin):
    def __init__(self, brown_dict=None):
        self.cue_pattern = re.compile('^' + '$|^'.join(filters.negation_cues) + '$', re.IGNORECASE)

    def predict(self, X):
        return [[bool(re.match(self.cue_pattern, token)) for token in tweet] for tweet in X]
