from storage.neg_cacher import NegCacher
from storage import data
from storage.options import General

def evaluate():
    train, test = data.get_data(General.TRAIN_SET, General.TEST_SET)
    test_tweets = [tweet.decode("utf-8") for tweet in test[:,0][:10]]
    NegCacher.cache(test_tweets)
    for tweet in test_tweets:
        print tweet
        print NegCacher.cached[tweet]
        print

if __name__ == "__main__":
    evaluate()
