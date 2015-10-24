from negation_classifier.crf_estimator import CRF, get_classifier
from negation_classifier.crf_transformer import CRFTransformer
from negation_classifier.negation_scope.twitter_negation_reader import parse_twitter_negation
from tweebo_cache import TweeboCacher


def _split_into_contexts(raw_tweets):
    crf_transformer = CRFTransformer()
    tweet_token_data = [[(token, pos, dep) for token, pos, dep in zip(
        TweeboCacher.get_cached_tokens()[tweet],
        TweeboCacher.get_cached_pos_tokens()[tweet],
        TweeboCacher.get_cached_dependency()[tweet]
    )] for tweet in raw_tweets]
    crf_data = crf_transformer.transform(tweet_token_data)

    predicted = get_classifier(parse_twitter_negation()).predict(crf_data)

    tweets = []
    for i, labels in enumerate(predicted):
        tweet = []
        first = True
        for j, label in enumerate(labels):
            token = TweeboCacher.get_cached_tokens()[raw_tweets[i]][j].lower()
            if label == 'negated':
                token += '_NEGFIRST' if first else '_NEG'
                first = False
            else:
                first = True
            tweet.append(token)
        tweets.append(tweet)
    return tweets


def _split_into_contexts_naive(raw_tweets):
    tweets = []
    for tweet in raw_tweets:
        contexts = []
        negated = False
        first = False
        for token in TweeboCacher.get_cached_tokens()[tweet]:
            token = token.lower()
            if token == 'not':
                negated = True
                first = True
            elif token[0] in (',', '.', ':', ';', '!', '?'):
                negated = False
            elif negated:
                token += '_NEGFIRST' if first else '_NEG'
                first = False
            contexts.append(token)
        tweets.append(contexts)
    return tweets

class NegCacher(object):
    cached = {}

    @staticmethod
    def cache(raw_tweets):
        naive = False
        if any([tweet not in NegCacher.cached for tweet in raw_tweets]):
            cached = _split_into_contexts_naive(raw_tweets) if naive else _split_into_contexts(raw_tweets)
            for raw_tweet, cached_tweet in zip(raw_tweets, cached):
                NegCacher.cached[raw_tweet] = cached_tweet
