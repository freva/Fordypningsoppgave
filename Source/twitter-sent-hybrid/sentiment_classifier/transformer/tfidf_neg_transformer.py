from sklearn.feature_extraction.text import TfidfVectorizer
from utils import filters as f


class TfidfNegTransformer(TfidfVectorizer):
    def __init__(self, negation_scope_length=None, preprocessors=[], tokenizer=None, analyzer='word', ngram_range=(1, 1), max_df=1.0,
                 min_df=1, use_idf=True, smooth_idf=True, sublinear_tf=False):
        super(TfidfNegTransformer, self).__init__(self, tokenizer=tokenizer, analyzer=analyzer, ngram_range=ngram_range,
            max_df=max_df, min_df=min_df, use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)
        self.negation_scope_length = negation_scope_length
        self.preprocessors = preprocessors

    def filter(self, docs):
        filtered_tweets = []
        for tweet in docs:
            for preprocessor in self.preprocessors:
                tweet = preprocessor(tweet)

            if self.negation_scope_length != None:
                tweet = TfidfNegTransformer.split_into_contexts_naive2(tweet, self.negation_scope_length)
            filtered_tweets.append(tweet)
        return filtered_tweets

    def fit(self, raw_documents, y=None):
        return super(TfidfNegTransformer, self).fit(self.filter(raw_documents), y)

    def fit_transform(self, raw_documents, y=None):
        return super(TfidfNegTransformer, self).fit_transform(self.filter(raw_documents), y)

    def transform(self, raw_documents, **kwargs):
        return super(TfidfNegTransformer, self).transform(self.filter(raw_documents), **kwargs)


    @staticmethod
    def split_into_contexts_naive2(tweet, limit_negation_length):
        contexts = []
        in_scope, negated = False, 0
        for token in tweet.split():
            token = token.lower()
            if token in f.negation_cues:
                in_scope = True
                negated = limit_negation_length
            elif token[0] in f.punctuation:
                in_scope = False
            elif in_scope and negated != 0:
                token += '_NEG'
                negated -= 1
            contexts.append(token)
        return ' '.join(contexts)
