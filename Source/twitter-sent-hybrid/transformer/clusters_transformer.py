from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
from utils import tokenizer
from storage import resource_reader


class ClusterTransformer(TransformerMixin, BaseEstimator):
    dictionary = None

    def __init__(self, norm=True, preprocessors=None):
        ClusterTransformer.dictionary = resource_reader.get_brown_cluster_dict()
        self.cluster_dict = dict.fromkeys(ClusterTransformer.dictionary.values(), 0)
        self.normalize = norm
        self.vectorizer = None
        self.preprocessors = preprocessors

    def fit(self, raw_tweets, y=None):
        self.vectorizer = DictVectorizer().fit([self.cluster_dict])
        return self

    def transform(self, raw_tweets):
        occurrence_list = []
        for tweet in raw_tweets:
            for preprocessor in self.preprocessors:
                tweet = preprocessor(tweet)

            occurrences = self.cluster_dict.copy()
            for token in tokenizer.tokenize(tweet):
                if token in ClusterTransformer.dictionary:
                    occurrences[ClusterTransformer.dictionary[token]] += 1
            occurrence_list.append(occurrences)
        vectorized = self.vectorizer.transform(occurrence_list)
        return normalize(vectorized, axis=0) if self.normalize else vectorized