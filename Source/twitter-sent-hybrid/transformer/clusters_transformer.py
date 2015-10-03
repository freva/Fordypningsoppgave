from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
from utils import tokenizer

class ClusterTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, dictionary, norm=True, preprocessor=None):
        self.brown_dict = dictionary()
        self.cluster_dict = dict.fromkeys(self.brown_dict.values(), 0)
        self.normalize = norm
        self.vectorizer = None
        self.preprocessor = preprocessor


    def fit(self, raw_tweets, y=None):
        self.vectorizer = DictVectorizer().fit([self.cluster_dict])
        return self


    def transform(self, raw_tweets):
        occurrence_list = []  # list holding occurrence dicts for each tweet
        for tweet in raw_tweets:
            if self.preprocessor:
                tweet = self.preprocessor(tweet)
            occurrences = self.cluster_dict.copy()
            for token in tokenizer.tokenize(tweet):
                if token in self.brown_dict:
                    occurrences[self.brown_dict[token]] += 1
            occurrence_list.append(occurrences)
        vectorized = self.vectorizer.transform(occurrence_list)
        return normalize(vectorized, axis=0) if self.normalize else vectorized