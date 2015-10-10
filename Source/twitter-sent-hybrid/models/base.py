"""
    Base class for different methods of using sentiment analysis.
"""
from sklearn.pipeline import Pipeline

from storage import cache
import utils.preprocessor_methods as pr


class BaseMethod(object):
    def __init__(self, train, feature_union=None, useCache=True, clf=None, defaults={}):
        self.options = {
            'vect__ngram_range': [(1, 1)],  # (2, 2), (3,3)],
            #'vect__stop_words': ('english', None),
            'vect__preprocessor': (
            None, pr.no_prep, pr.no_usernames, pr.remove_noise, pr.placeholders, pr.all, pr.remove_all,
            pr.reduced_attached, pr.no_url_username_reduced_attached),
            'vect__use_idf': (True, False),
            'vect__max_df': (0.5,),
            'vect__smooth_idf': (True, False),
            'vect__sublinear_tf': (True, False)
        }

        self.clf = clf(**defaults)
        self.useCache = useCache
        self.feature_union = feature_union
        self.train(train[:,0], train[:,1])


    def train(self, docs_train, y_train):
        self.grid = Pipeline([
            ('features', self.feature_union),
            ('clf', self.clf)
        ])

        #vars = [model[1] for model in pipeline.steps[0][1].transformer_list]
        #print [var.fit_transform(docs_train, y_train).shape for var in vars]

        cache_key = str(self.feature_union) + str(docs_train)
        cached = cache.get(cache_key)

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

            cache.save(cache_key, {
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
