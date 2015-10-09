"""
    Sentiment analysis using basic bigrams.
"""
from models.base import BaseMethod
from utils import utils
import numpy as np

class Combined:
    def __init__(self, sub_clf_options, pol_clf_options, train):
        train_subjectivity, train_polarity = utils.generate_two_part_dataset(train)
        self.subjectivity_clf = BaseMethod(train_subjectivity, **sub_clf_options)
        self.polarity_clf = BaseMethod(train_polarity, **pol_clf_options)

        self.best_score = (self.subjectivity_clf.best_score + self.polarity_clf.best_score) / 2


    def predict(self, arg_input):
        orig = arg_input
        if isinstance(arg_input, basestring):
            orig = [orig]

        subjective_predictions = self.subjectivity_clf.predict(orig)
        temp = np.array([[i, orig[i]] for i, sub_pred in enumerate(subjective_predictions) if sub_pred != "neutral"])
        pol_ids, pol_docs = list(int(i) for i in temp[:,0]), temp[:,1]

        polarity_predictions = self.polarity_clf.predict(pol_docs)
        for i, pol_id in enumerate(pol_ids):
            subjective_predictions[pol_id] = polarity_predictions[i]
        return subjective_predictions
