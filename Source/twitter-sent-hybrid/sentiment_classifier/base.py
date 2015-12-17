from sklearn.pipeline import Pipeline
from storage import cache
import re


class BaseMethod(object):
    def __init__(self, train, feature_union=None, useCache=True, clf=None, defaults={}):
        self.clf = clf(**defaults)
        self.useCache = useCache
        self.feature_union = feature_union
        self.grid = Pipeline([
            ('features', self.feature_union),
            ('clf', self.clf)
        ])
        self.train(train[:, 0], train[:, 1])

    def train(self, docs_train, y_train):
        feature_union_hash = re.sub(r" at 0x[0-9a-f]+>", "", str(self.feature_union.transformer_list))
        cache_key = feature_union_hash + str(self.clf) + str(docs_train)
        cached = cache.load_pickle(cache_key)
        
        if cached and self.useCache:
            print "Loading from cache..."
            self.best_estimator = cached['est']
            self.best_score = cached['scr']
            self.best_params = cached['parm']
        else:
            self.grid.fit(docs_train, y_train)

            self.best_estimator = self.grid
            self.best_params = self.grid.get_params(False)
            self.best_score = 1

            cache.save_pickle(cache_key, {
                "est": self.best_estimator,
                "scr": self.best_score,
                "parm": self.best_params
            })
        return self.grid

    def predict(self, arg_input):
        orig = arg_input
        if isinstance(arg_input, basestring):
            orig = [orig]

        predictions = self.best_estimator.predict(orig)
        if isinstance(arg_input, basestring):
            return predictions[0]
        return predictions
