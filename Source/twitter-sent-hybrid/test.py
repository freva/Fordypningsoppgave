from sklearn.metrics import classification_report
from negation_classifier.negation_scope.twitter_negation_reader import parse_twitter_negation
from negation_classifier.cue_detector import CueDetector
from storage.neg_cacher import NegCacher
from storage import data
from storage.options import General
from negation_classifier import crf_estimator

def evaluate():
    # _, brown_dict = load_clusters()
    train, test = data.get_data(General.TRAIN_SET, General.TEST_SET)
    test_tweets = [tweet.decode("utf-8") for tweet in test[:,0][:10]]
    NegCacher.cache(test_tweets)
    for tweet in test_tweets:
        print tweet
        print NegCacher.cached[tweet]
        print



if __name__ == "__main__":
    evaluate()
