from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import nltk
import utils.preprocessor_methods as t


class POSTransformer(TransformerMixin, BaseEstimator):
    classes = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB"]
    def __init__(self, norm=True):
        self.vectorizer = None
        self.normalize = norm
        self.pos_dict = dict.fromkeys(POSTransformer.classes, 0)

    def transform(self, raw_tweets, **transform_params):
        occurences = []
        for tweet in raw_tweets:
            tweet = t.remove_all(tweet)
            tokens = nltk.word_tokenize(tweet)
            tagged = nltk.pos_tag(tokens)
            tag_frequencies = self.pos_dict.copy()
            for word, tag in tagged:
                if tag in tag_frequencies:
                    tag_frequencies[tag] += 1
            occurences.append(tag_frequencies)
        vectorized = self.vectorizer.transform(occurences)
        return normalize(vectorized) if self.normalize else vectorized

    def fit(self, raw_tweets, y=None):
        self.vectorizer = DictVectorizer().fit([self.pos_dict])
        return self
