"""
    A POST Server running a sentimental analysis on a tweet
    stringified JSON Object.

    Takes POST requests and returns a string with the classification.
    Classification scheme: <neutral, positive, negative>
"""
import sys

# System specific
import storage.data as d
import utils.preprocessor_methods as pr

# Do import of all different methods here:
# Remember: When adding a new method, add it to the methods/__init__.py
from models import *

d.set_file_names()
docs_test, y_test, docs_train, y_train, docs_train_subjectivity, y_train_subjectivity, docs_train_polarity, y_train_polarity = d.get_data()

c1_vect_options = {
  'ngram_range': (1,1),
  'sublinear_tf': True,
  'preprocessor': pr.remove_noise,
  'use_idf': True,
  'smooth_idf': True,
  'max_df': 0.5
}

c1_default_options = {'C': 0.3}

clf = SVM(docs_train, y_train, default_options=c1_default_options, vect_options=c1_vect_options)

text = ' '.join(sys.argv[1:])
print(clf.predict(text))
