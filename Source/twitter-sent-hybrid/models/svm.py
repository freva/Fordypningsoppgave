"""
    Sentiment analysis using basic bigrams.
"""
from sklearn.svm import LinearSVC
from base import BaseMethod
from storage.options import Feature


class SVM(BaseMethod):
    def __init__(self, docs_train, y_train, options, useCrossValidation=False):
        self.clf = LinearSVC(**options[Feature.SVM_DEFAULT_OPTIONS])

        extra = {
            'clf__C': (0.1, 0.3, 0.5, 0.7, 0.8, 1.0,),
        }
        super(SVM, self).__init__(docs_train, y_train, options, extra, useCrossValidation)
