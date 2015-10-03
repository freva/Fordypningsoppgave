"""
    Sentiment analysis using basic bigrams.
"""
from sklearn.svm import LinearSVC
from base import BaseMethod
from storage.options import Feature


class SVM(BaseMethod):
    def __init__(self, docs_train, y_train, useCrossValidation=False):
        self.clf = LinearSVC(**Feature.SVM_DEFAULT_OPTIONS)

        super(SVM, self).__init__(docs_train, y_train, useCrossValidation)
