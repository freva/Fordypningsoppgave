from sklearn.metrics import classification_report
from negation_classifier.negation_scope.twitter_negation_reader import parse_twitter_negation
from negation_classifier.cue_detector import CueDetector

def evaluate():
    # _, brown_dict = load_clusters()
    detector = CueDetector()
    tweets = parse_twitter_negation()
    tokens = [[token for token, _, _, _, _ in tweet]
              for tweet in tweets]
    cues = [is_cue for tweet in tweets for _, _, _, is_cue, _ in tweet]
    pred_cues = [is_cue for tweet in detector.predict(tokens) for is_cue in tweet]
    print(classification_report(cues, pred_cues, digits=3))


if __name__ == "__main__":
    evaluate()
