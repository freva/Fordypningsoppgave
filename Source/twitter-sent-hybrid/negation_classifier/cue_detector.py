import re
from sklearn.base import BaseEstimator, ClassifierMixin


class CueDetector(BaseEstimator, ClassifierMixin):
    def __init__(self, brown_dict=None):
        negation_cues = ['hardly', 'lack', 'lacking', 'lacks', 'neither', 'nor', 'never', 'no', 'nobody', 'none',
                         'nothing', 'nowhere', 'not', 'without', 'aint', 'cant', 'cannot', 'darent', 'dont', 'doesnt',
                         'didnt', 'hadnt', 'hasnt', 'havent', 'havnt', 'isnt', 'mightnt', 'mustnt', 'neednt', 'oughtnt',
                         'shant', 'shouldnt', 'wasnt', 'wouldnt', ".*n't"]
        if brown_dict:
            negation_cues.extend(self.get_clusters(negation_cues, brown_dict))
        self.cue_pattern = re.compile('^' + '$|^'.join(negation_cues) + '$', re.IGNORECASE)

    def predict(self, X):
        return [[bool(re.match(self.cue_pattern, token)) for token in tweet] for tweet in X]

    def predict_token(self, token):
        return re.match(self.cue_pattern, token)

    @staticmethod
    def get_clusters(negation_cues, brown_dict):
        identified = []
        for word in negation_cues:
            print("current word: " + word)
            try:
                word_id = brown_dict[word]
                for key, value in brown_dict.items():
                    if value == word_id:
                        identified.append(re.escape(key))
            except KeyError:
                print(word)
        return identified

