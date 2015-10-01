import re

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import normalize
import numpy as np
from nltk.tokenize import wordpunct_tokenize




class PunctuationTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, norm=True):
        self.normalize = norm

    def fit(self, raw_tweets, y=None):
        return self

    def transform(self, raw_tweets):
        vectorized = np.zeros((len(raw_tweets), 4))
        for i, tweet in enumerate(raw_tweets):
            exclamations = 0
            questions = 0
            both = 0
            wordArray = wordpunct_tokenize(tweet)
            for word in wordArray:
                word = re.sub(r"[,.]", "", word)
                if re.match(r"\!{2,}$", word):
                    exclamations += 1
                if re.match(r"\?{2,}$", word):
                    questions += 1
                if re.match(r"(?:\!+\?+)|(?:\?+\!+)[?!]*", word):
                    both += 1
            vectorized[i][0] = exclamations
            vectorized[i][1] = questions
            vectorized[i][2] = both
            vectorized[i][3] = 1 if re.match(r"[!?]{2,}", wordArray[-1]) else 0
        return normalize(vectorized) if self.normalize else vectorized


def main():
    pt = PunctuationTransformer()
    pt.transform(["shit!!!! why?? meee???"])

if __name__ == '__main__':
    main()


