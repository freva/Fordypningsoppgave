import re
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import normalize
from storage import resource_reader


class LexiconTransformer(TransformerMixin, BaseEstimator):
    negated_RE = re.compile(r'(.*)_NEG(?:FIRST)?$')
    manual_lexicon = [resource_reader.get_bing_liu_lexicon(), resource_reader.get_mpqa_lexicon(), resource_reader.get_nrc_emotion_lexicon(), resource_reader.get_afinn_lexicon()]
    automatic_lexicon = [(resource_reader.get_automated_lexicon(name), bigram) for name, bigram in [
        ('../data/lexica/Sentiment140/S140-AFFLEX-NEGLEX-unigrams.txt',      False),
        ('../data/lexica/Sentiment140/S140-AFFLEX-NEGLEX-bigrams.txt',       True),
        ('../data/lexica/HashtagSentiment/HS-AFFLEX-NEGLEX-unigrams.txt',    False),
        ('../data/lexica/HashtagSentiment/HS-AFFLEX-NEGLEX-bigrams.txt',     True)
    ]]

    def __init__(self, norm=True, preprocessors=None):
        self.normalize = norm
        self.preprocessors = preprocessors

    def fit(self, raw_tweets, y=None):
        return self

    def transform(self, raw_tweets, y=None):
        filtered_tweets = []
        for tweet in raw_tweets:
            for preprocessor in self.preprocessors:
                tweet = preprocessor(tweet)
            filtered_tweets.append(tweet)

        matrix = self._automatic_lexicon_scorer(filtered_tweets, LexiconTransformer.automatic_lexicon[0][0], LexiconTransformer.automatic_lexicon[0][1])
        for lexicon in LexiconTransformer.automatic_lexicon[1:]:
            matrix = np.concatenate((matrix, self._automatic_lexicon_scorer(filtered_tweets, lexicon[0], lexicon[1])), axis=1)

        for lexicon in LexiconTransformer.manual_lexicon:
            matrix = np.concatenate((matrix, self._manual_lexicon_scorer(filtered_tweets, lexicon)), axis=1)
        return matrix


    def _automatic_lexicon_scorer(self, raw_tweets, lexicon, bigram):
        scores = []
        for contexts in raw_tweets:
            tweet_scores = []
            contexts = contexts.split(" ")
            if bigram:
                contexts = zip(contexts, contexts[1:])
            for token in contexts:
                if bigram:
                    word_1 = token[0]
                    word_2 = token[1]
                    if LexiconTransformer.negated_RE.match(token[0]):
                        word_1 = LexiconTransformer.negated_RE.sub(r'\1', token[0])
                    elif LexiconTransformer.negated_RE.match(token[1]):
                        word_2 = LexiconTransformer.negated_RE.sub(r'\1', token[1])
                    token = (word_1, word_2)
                    token = ' '.join(token)
                else:
                    if LexiconTransformer.negated_RE.match(token):
                        token = LexiconTransformer.negated_RE.sub(r'\1', token)
                tweet_scores.append(lexicon[token] if token in lexicon else 0)

            scores.append([len([score for score in tweet_scores if score != 0]),
                           len(tweet_scores),
                           sum(tweet_scores) if tweet_scores else 0,
                           max(map(abs, tweet_scores)) if tweet_scores else 0,
                           tweet_scores[-1] if tweet_scores else 0])
        return normalize(scores) if self.normalize else scores


    def _manual_lexicon_scorer(self, raw_tweets, lexicon_dict):
        scores = np.zeros((len(raw_tweets), 4))
        for i, contexts in enumerate(raw_tweets):
            for token in contexts.split(" "):
                try:
                    if LexiconTransformer.negated_RE.match(token):
                        token = LexiconTransformer.negated_RE.sub(r'\1', token)
                        scores[i][2 if lexicon_dict[token] > 0 else 3] += lexicon_dict[token]
                    else:
                        scores[i][0 if lexicon_dict[token] > 0 else 1] += lexicon_dict[token]
                except KeyError:
                    pass
        return normalize(scores) if self.normalize else scores


