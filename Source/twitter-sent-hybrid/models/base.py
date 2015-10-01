"""
    Base class for different methods of using sentiment analysis.
"""
import sys

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import FeatureUnion

import utils.preprocessor_methods as pr
import utils.tokenizer as t
from storage import cache
from transformer import ClusterTransformer
from transformer import AllcapsTransformer
from transformer import WordCounter
from storage.options import Feature


class BaseMethod(object):
    def __init__(self, docs_train, y_train, options, extra={}, useCrossValidation=False, vect_options={}):
        if sys.flags.debug:
            self.options = {}
        else:
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


        pipeline = Pipeline([
            ('features', FeatureUnion([
                (Feature.WORD_VECTORIZER, TfidfVectorizer(tokenizer=t.tokenize, **options[Feature.WORD_VECTORIZER])),
                (Feature.CHAR_NGRAMS, TfidfVectorizer(analyzer='char', **options[Feature.CHAR_NGRAMS])),
                (Feature.WORD_CLUSTERS, ClusterTransformer(**options[Feature.WORD_CLUSTERS])),
                (Feature.ALLCAPS, AllcapsTransformer()),
                ('count', WordCounter())
            ])),
            ('clf', self.clf)
        ])

        #vars = [model[1] for model in pipeline.steps[0][1].transformer_list]
        #print [var.fit_transform(docs_train, y_train).shape for var in vars]

        useGrid = sys.flags.optimize
        if useGrid:   #TODO: fix grid search
            self.grid = GridSearchCV(
                pipeline,
                options,
                cv=cv,
                refit=True,
                n_jobs=-1,
                verbose=1
            )
        else:
            self.grid = pipeline

        cache_key = str(self.grid) + str(docs_train)
        cached = cache.get(cache_key)

        if cached and sys.flags.debug == 0:
            self.best_estimator = cached['est']
            self.best_score = cached['scr']
            self.best_params = cached['parm']

        self.grid.fit(docs_train, y_train)

        if useGrid:
            self.best_estimator = self.grid.best_estimator_
            self.best_params = self.grid.best_params_
            self.best_score = self.grid.best_score_
        else:
            self.best_estimator = self.grid
            self.best_params = self.grid.get_params(False)
            self.best_score = 1

            cache.save(cache_key, {
                "est": self.best_estimator,
                "scr": self.best_score,
                "parm": self.best_params
            })

        self.steps = self.best_estimator.named_steps
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
