from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
from storage import cache
import nltk


class POSTransformer(TransformerMixin, BaseEstimator):
    classes = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT",
               "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP",
               "VBZ", "WDT", "WP", "WP$", "WRB"]
    cached = cache.get("dict", "pos_frequencies_")
    occurrences = {} if not cached else cached

    def __init__(self, norm=True, preprocessor=None):
        self.vectorizer = None
        self.normalize = norm
        self.preprocessor = preprocessor
        self.pos_dict = dict.fromkeys(POSTransformer.classes, 0)


    def fit(self, raw_tweets, y=None):
        self.vectorizer = DictVectorizer().fit([self.pos_dict])
        return self


    def transform(self, raw_tweets, **transform_params):
        updated = False
        pos_tabs = []

        for raw_tweet in raw_tweets:
            if raw_tweet in POSTransformer.occurrences:
                tag_frequencies = POSTransformer.occurrences[raw_tweet]
            else:
                tweet = self.preprocessor(raw_tweet) if self.preprocessor else raw_tweet
                tagged = nltk.pos_tag(nltk.word_tokenize(tweet))
                tag_frequencies = self.pos_dict.copy()

                for word, tag in tagged:
                    if tag in tag_frequencies:
                        tag_frequencies[tag] += 1
                POSTransformer.occurrences[raw_tweet] = tag_frequencies
                updated = True
            pos_tabs.append(tag_frequencies)

        if updated:
            cache.save("dict", POSTransformer.occurrences, "pos_frequencies_")
        vectorized = self.vectorizer.transform(pos_tabs)
        return normalize(vectorized) if self.normalize else vectorized
