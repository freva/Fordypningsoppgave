from negation_classifier.crf_estimator import CRF, get_classifier
from negation_classifier.crf_transformer import CRFTransformer
from tweebo_cache import TweeboCacher
from storage import cache
from storage import resource_reader


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


def parse_twitter_negation():
    tweets = resource_reader.get_twitter_negation_corpus()

    pos_tokens = cache.load_json("neg_pos_tokens_cache")
    dependency_tweets = cache.load_json("neg_dependency_cache")

    return [[(token, pos_tokens[str(i)][j], dependency_tweets[str(i)][j], is_cue, label)
            for j, (token, label, is_cue) in enumerate(sentence)] for i, sentence in enumerate(tweets)]


class NegCacher(object):
    cached = {}

    @staticmethod
    def cache(raw_tweets):
        naive = False
        if any([tweet not in NegCacher.cached for tweet in raw_tweets]):
            cached = _split_into_contexts_naive(raw_tweets) if naive else _split_into_contexts(raw_tweets)
            for raw_tweet, cached_tweet in zip(raw_tweets, cached):
                NegCacher.cached[raw_tweet] = cached_tweet
