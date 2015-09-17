#!/usr/bin/python
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

import storage.data as d


def translate_to_numbers(li):
    return [conv[v] for v in li]


conv = {
    '"negative"': 0,
    '"neutral"': 1,
    '"objective"': 2,
    '"objective-OR-neutral"': 3,
    '"positive"': 4
}

headers = ['"negative"', '"neutral"', '"objective"', '"objective-OR-neutral"', '"positive"']

train_set_filename = '../Testing/2013-2-train-full-B.tsv'
test_set_filename = '../Testing/2013-2-test-gold-B.tsv'

train, test = d.set_file_names(train_set=train_set_filename, test_set=test_set_filename)

docs_train = train[:, 3]
y_train = train[:, 2]

docs_test = test[:, 3]
y_test = test[:, 2]

vect = CountVectorizer()
pipeline = Pipeline([
    ('vect', vect),
    ('tfidf', TfidfTransformer()),
    ('clf', LinearSVC()),
])

cv = StratifiedKFold(y_train, n_folds=5)

grid = GridSearchCV(
    pipeline,
    {
        # 'vect__ngram_range': ((1, 1), (2, 2), (3,3)),
        # 'vect__ngram_range': ((1, 1), (2, 2)),
        # 'vect__stop_words': ('english', None),
        # 'vect__preprocessor': (None, pr.no_usernames, pr.remove_noise, pr.placeholders, pr.all),
        # 'vect__preprocessor': (None, pr.no_usernames, pr.all),
        'tfidf__use_idf': (True, False),
        # 'tfidf__smooth_idf': (True, False),
        # 'tfidf__sublinear_tf': (True, False),
        # 'clf__alpha': tuple( np.arange(0, 1.0, 0.1) ),
        # 'clf__C': ( 100, ),
    },
    # cv=cv,
    refit=True,
    n_jobs=-1,
    verbose=1
)

print "Training "
grid.fit(docs_train, y_train)
print "Done training"

print "\nBest Params:"
print grid.best_params_

y_predicted = grid.best_estimator_.predict(docs_test)

# Translate classifications to numbers to allow classification_report
y_test_int = translate_to_numbers(y_test)
y_predicted_int = translate_to_numbers(y_predicted)

print "\nClassification Report:"
print classification_report(y_test_int, y_predicted_int, target_names=headers)

# Plot the confusion matrix
print "\nConfusion matrix: "
print confusion_matrix(y_test_int, y_predicted_int)

print "\nBest score: %s" % grid.best_score_
print "\nCalculated acc, np.mean(y_predicted == y_test): %s" % np.mean(y_predicted == y_test)
print "Accuracy Score, grid.score(docs_test, y_test): %.2f" % grid.score(docs_test, y_test)
