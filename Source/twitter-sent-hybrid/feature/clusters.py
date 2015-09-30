from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
from utils import tokenizer

class ClusterTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, cluster_dict, brown_dict, norm=True):
        self.brown_dict = brown_dict
        self.cluster_dict = cluster_dict
        self.normalize = norm
        self.vectorizer = None
 
    def fit(self, raw_tweets, y=None):
        self.vectorizer = DictVectorizer().fit([self.cluster_dict])
        return self
 
    def transform(self, raw_tweets):
        occurrence_list = []  # list holding occurrence dicts for each tweet
        for tweet in raw_tweets:
            occurrences = self.cluster_dict.copy()
            for token in tokenizer.tokenize(tweet):
                if token in self.brown_dict:
                    occurrences[self.brown_dict[token]] += 1
            occurrence_list.append(occurrences)
        vectorized = self.vectorizer.transform(occurrence_list)
        return normalize(vectorized, axis=0) if self.normalize else vectorized