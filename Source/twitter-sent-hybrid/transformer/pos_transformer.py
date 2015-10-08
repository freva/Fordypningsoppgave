from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import nltk
import utils.preprocessor_methods as t


class POSTransformer(TfidfVectorizer):
    def __init__(self, norm=False, **kwargs):
        super(POSTransformer, self).__init__(self, **kwargs)
        self.vectorizer = None
        self.normalize = norm

    def transform(self, raw_tweets, **transform_params):
        pos_tweets = []
        for tweet in raw_tweets:
            tweet = t.remove_all(tweet)
            tokens = nltk.word_tokenize(tweet)
            tagged = nltk.pos_tag(tokens)
            new_sentence = " ".join([i[1] for i in tagged])
            pos_tweets.append(new_sentence)
        #vectorized = self.vectorizer.transform(pos_tweets)
        self.X = normalize(pos_tweets) if self.normalize else pos_tweets
        return self.X
