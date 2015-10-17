from sklearn.feature_extraction.text import TfidfVectorizer

class TfidfNegTransformer(TfidfVectorizer):
    def __init__(self, negate=True, preprocessors=[], **kwargs):
        super(TfidfNegTransformer, self).__init__(self, **kwargs)
        self.negate = negate
        self.preprocessors = preprocessors

    def filter(self, docs):
        filtered_tweets = []
        for tweet in docs:
            for preprocessor in self.preprocessors:
                tweet = preprocessor(tweet)
            filtered_tweets.append(tweet)
        return filtered_tweets

    def fit(self, raw_documents, y=None):
        return super(TfidfNegTransformer, self).fit(self.filter(raw_documents), y)

    def fit_transform(self, raw_documents, y=None):
        return super(TfidfNegTransformer, self).fit_transform(self.filter(raw_documents), y)

    def transform(self, raw_documents, **kwargs):
        return super(TfidfNegTransformer, self).transform(self.filter(raw_documents), **kwargs)

