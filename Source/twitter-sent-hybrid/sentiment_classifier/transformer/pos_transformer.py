from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
from storage import pos_cache


class POSTransformer(TransformerMixin, BaseEstimator):
    classes = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT",
               "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP",
               "VBZ", "WDT", "WP", "WP$", "WRB"]

    def __init__(self, norm=True, preprocessors=[]):
        self.vectorizer = None
        self.normalize = norm
        self.preprocessors = preprocessors
        self.pos_dict = dict.fromkeys(POSTransformer.classes, 0)


    def fit(self, raw_tweets, y=None):
        self.vectorizer = DictVectorizer().fit([self.pos_dict])
        return self


    def transform(self, raw_tweets, **transform_params):
        pos_tabs = []

        for preprocessor in self.preprocessors:
            raw_tweets = [preprocessor(tweet) for tweet in raw_tweets]
        pos_tags = pos_cache.get_pos_tags(raw_tweets)

        for pos_tag in pos_tags:
            tag_frequencies = self.pos_dict.copy()

            for tag in pos_tag:
                if tag in tag_frequencies:
                    tag_frequencies[tag] += 1
            pos_tabs.append(tag_frequencies)

        vectorized = self.vectorizer.transform(pos_tabs)
        return normalize(vectorized) if self.normalize else vectorized