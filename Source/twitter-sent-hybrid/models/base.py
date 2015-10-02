"""
    Base class for different methods of using sentiment analysis.
"""
import sys

from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV

import utils.preprocessor_methods as pr
from storage import cache
from storage.options import Feature


class BaseMethod(object):
    def __init__(self, docs_train, y_train, options, extra={}, useCrossValidation=False, vect_options={}):
        self.options = {
            'vect__ngram_range': [(1, 1)],  # (2, 2), (3,3)],
            #'vect__stop_words': ('english', None),
            'vect__preprocessor': (
            None, pr.no_prep, pr.no_usernames, pr.remove_noise, pr.placeholders, pr.all, pr.remove_all,
            pr.reduced_attached, pr.no_url_usernames_reduced_attached),
            'vect__use_idf': (True, False),
            'vect__max_df': (0.5,),
            'vect__smooth_idf': (True, False),
            'vect__sublinear_tf': (True, False)
        }

        self.train(docs_train, y_train, options, extra, useCrossValidation, vect_options)


    def train(self, docs_train, y_train, options, extra={}, useCrossValidation=False, vect_options={}):
        #options = dict(self.options.items() + extra.items())
        cv = StratifiedKFold(y_train, n_folds=10) if useCrossValidation else None

        self.grid = Pipeline([
            ('features', Feature.feature_union),
            ('clf', self.clf)
        ])

        #vars = [model[1] for model in pipeline.steps[0][1].transformer_list]
        #print [var.fit_transform(docs_train, y_train).shape for var in vars]

        cache_key = str(options) + str(docs_train)
        cached = cache.get(cache_key)

        if cached:
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


    def __str__(self):
        return "%s" % self.__class__.__name__
