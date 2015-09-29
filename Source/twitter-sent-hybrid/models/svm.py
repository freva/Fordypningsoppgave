"""
    Sentiment analysis using basic bigrams.
"""
from sklearn.svm import LinearSVC
from base import BaseMethod


class SVM(BaseMethod):
    def __init__(self, docs_train, y_train,cluster_dict, brown_dict, useCrossValidation=False, default_options={}, vect_options={}):
        self.clf = LinearSVC(**default_options)

        extra = {
            'clf__C': (0.1, 0.3, 0.5, 0.7, 0.8, 1.0,),
        }
        super(SVM, self).__init__(docs_train, y_train,cluster_dict, brown_dict, extra, useCrossValidation, vect_options=vect_options)
