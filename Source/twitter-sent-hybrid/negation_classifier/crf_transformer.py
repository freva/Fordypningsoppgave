import pycrfsuite
from sklearn.base import BaseEstimator, TransformerMixin
from cue_detector import CueDetector


class CRFTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, max_distance=5):
        self.max_distance = max_distance
        self.cue_detector = CueDetector()

    def fit(self, X, y=None):
        return self

    def transform(self, tokenized_tweets, tweet_cues=None):
        if not tweet_cues:
            tweet_cues = self.cue_detector.predict(
                [[token for token, _, _ in tweet_data] for tweet_data in tokenized_tweets]
            )
        return [pycrfsuite.ItemSequence(self._tweet_to_features(tweet, tweet_cues[i]))
                for i, tweet in enumerate(tokenized_tweets)]

    def _token_to_features(self, tweet, tweet_cues, i):
        token = tweet[i][0]
        pos_tag = tweet[i][1]
        self.tweet_cues = tweet_cues
        self.dependencies_up = [dep for _, _, dep in tweet]
        self.dependencies_down = {}
        for j, dep in enumerate(self.dependencies_up):
            try:
                self.dependencies_down[dep].append(j)
            except KeyError:
                self.dependencies_down[dep] = [j]
        features = {
            'token.lower': token.lower(),
            'pos_tag': pos_tag,
            'right_distance': str(self._token_wise_distance(tweet_cues, i, True)),
            'left_distance': str(self._token_wise_distance(tweet_cues, i, False)),
        }

        dep1 = self.dependencies_up[i]
        if dep1 >= 0:
            features['dep:distance'] = str(self._dependency_distance(i, None, 0))
            pos_tag1 = tweet[dep1][1]
            features['dep1:pos_tag'] = pos_tag1
            features['dep1:distance'] = str(self._dependency_distance(dep1, None, 0))

            dep2 = self.dependencies_up[dep1]
            if dep2 >= 0:
                pos_tag2 = tweet[dep2][1]
                features['dep2:pos_tag'] = pos_tag2
                features['dep2:distance'] = str(self._dependency_distance(dep2, None, 0))

        # if i == 0:
        #     features['__BOS__'] = None
        # elif i == len(tweet) - 1:
        #     features['__EOS__'] = None

        return features

    def _tweet_to_features(self, tweet, tweet_cues):
        return [self._token_to_features(tweet, tweet_cues, i) for i in range(len(tweet))]

    def _token_wise_distance(self, tweet_cues, i, right):
        if tweet_cues[i]:
            return 0
        else:
            for j, is_cue in enumerate(tweet_cues[i + 1:] if right else reversed(tweet_cues[:i]), start=1):
                if j == self.max_distance:
                    return j
                elif is_cue:
                    return j
        return self.max_distance

    def _dependency_distance(self, here, came_from, depth):
        if self.tweet_cues[here]:
            return depth
        if depth == self.max_distance or here < 0:
            return self.max_distance
        distances_from_here = []
        if self.dependencies_up[here] != came_from:
            distances_from_here.append(self._dependency_distance(self.dependencies_up[here], here, depth + 1))
        try:
            for down in self.dependencies_down[here]:
                if down != came_from:
                    distances_from_here.append(self._dependency_distance(down, here, depth + 1))
        except KeyError:
            return self.max_distance
        return min(distances_from_here) if distances_from_here else self.max_distance
