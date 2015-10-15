from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
from storage import cache
import nltk


class POSTransformer(TransformerMixin, BaseEstimator):
    pos_tweets = cache.get("pos_tags", False)
    if not pos_tweets:
        raise Exception("PoS cache not found!")
    classes = ['BES', 'CC', 'CD', 'DT', 'EX', 'FW', 'HT', 'IN', 'JJ', 'JJR', 'JJS', 'MB', 'MD', 'NN', 'NNP', 'NNPS',
               'NNS', 'PDT', 'POS', 'PRP', 'RB', 'RBR', 'RBS', 'RP', 'RT', 'SYM', 'TO', 'UH', 'URL', 'USR', 'VB',
               'VBD','VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WRB', 'X']


    def __init__(self, norm=True, preprocessor=None):
        self.vectorizer = None
        self.normalize = norm
        self.preprocessor = preprocessor
        self.pos_dict = dict.fromkeys(POSTransformer.classes, 0)


    def fit(self, raw_tweets, y=None):
        self.vectorizer = DictVectorizer().fit([self.pos_dict])
        return self


    def transform(self, raw_tweets, **transform_params):
        pos_tabs = []

        for raw_tweet in raw_tweets:
            if raw_tweet in POSTransformer.pos_tweets:
                occurrences = self.pos_dict.copy()
                for pos in POSTransformer.pos_tweets[raw_tweet]:
                    occurrences[pos] += 1
                pos_tabs.append(occurrences)
            else:
                raise Exception("Tweet \"" + raw_tweet + "\" not found in cache")
        vectorized = self.vectorizer.transform(pos_tabs)
        return normalize(vectorized) if self.normalize else vectorized
