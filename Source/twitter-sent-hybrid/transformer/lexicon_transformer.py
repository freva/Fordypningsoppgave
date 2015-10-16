import re
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import normalize
from storage import lexicon


class LexiconTransformer(TransformerMixin, BaseEstimator):
    manual_lexicon = [lexicon.get_bing_liu_lexicon(), lexicon.get_mpqa_lexicon(), lexicon.get_nrc_emotion_lexicon()]
    automatic_lexicon = [(lexicon.get_automated_lexicon(name), bigram) for name, bigram in [
        ('../Testing/lexica/Sentiment140/S140-AFFLEX-NEGLEX-unigrams.txt',      False),
        ('../Testing/lexica/Sentiment140/S140-AFFLEX-NEGLEX-bigrams.txt',       True),
        ('../Testing/lexica/HashtagSentiment/HS-AFFLEX-NEGLEX-unigrams.txt',    False),
        ('../Testing/lexica/HashtagSentiment/HS-AFFLEX-NEGLEX-bigrams.txt',     True)
    ]]

    def __init__(self, norm=True, preprocessor=None):
        self.normalize = norm
        self.preprocessor = preprocessor


    def fit(self, raw_tweets, y=None):
        return self

    def transform(self, raw_tweets, y=None):
        matrix = self._automatic_lexicon_scorer(raw_tweets, LexiconTransformer.automatic_lexicon[0][0], LexiconTransformer.automatic_lexicon[0][1])
        for lexicon in LexiconTransformer.automatic_lexicon[1:]:
            matrix = np.concatenate((matrix, self._automatic_lexicon_scorer(raw_tweets, lexicon[0], lexicon[1])), axis=1)

        for lexicon in LexiconTransformer.manual_lexicon:
            matrix = np.concatenate((matrix, self._manual_lexicon_scorer(raw_tweets, lexicon)), axis=1)
        return matrix


    def _automatic_lexicon_scorer(self, raw_tweets, lexicon, bigram):
        scores = np.zeros((len(raw_tweets), 4))
        for i, contexts in enumerate(raw_tweets):
            tweet_scores = []
            contexts = (self.preprocessor(contexts)).split(" ") if self.preprocessor else contexts.split(" ")
            if bigram:
                contexts = zip(contexts, contexts[1:])
            for token in contexts:
                negated_regex = r'(.*)_NEG(?:FIRST)?$'
                if bigram:
                    word_1 = token[0]
                    word_2 = token[1]
                    if re.match(negated_regex, token[0]):
                        word_1 = re.sub(negated_regex, r'\1', token[0])
                    elif re.match(negated_regex, token[1]):
                        word_2 = re.sub(negated_regex, r'\1', token[1])
                    token = (word_1, word_2)
                    token = ' '.join(token)
                else:
                    if re.match(negated_regex, token):
                        token = re.sub(negated_regex, r'\1', token)
                try:
                    tweet_scores.append(lexicon[token])
                except KeyError:
                    pass
            scores[i][0] = len([score for score in tweet_scores if score != 0])
            scores[i][1] = sum(tweet_scores) if tweet_scores else 0
            scores[i][2] = max(map(abs, tweet_scores)) if tweet_scores else 0
            scores[i][3] = tweet_scores[-1] if tweet_scores else 0
        return normalize(scores) if self.normalize else scores


    def _manual_lexicon_scorer(self, raw_tweets, lexicon_dict):
        scores = np.zeros((len(raw_tweets), 4))
        for i, contexts in enumerate(raw_tweets):
            contexts = (self.preprocessor(contexts)).split(" ") if self.preprocessor else contexts.split(" ")
            for token in contexts:
                try:
                    negated_regex = r'(.*)_NEG(?:FIRST)?$'
                    if re.match(negated_regex, token):
                        token = re.sub(negated_regex, r'\1', token)
                        scores[i][2 if lexicon_dict[token] > 0 else 3] += lexicon_dict[token]
                    else:
                        scores[i][0 if lexicon_dict[token] > 0 else 1] += lexicon_dict[token]
                except KeyError:
                    pass
        return normalize(scores) if self.normalize else scores


