#!/usr/bin/python
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from utils import filters as f
from utils import preprocess as p

import storage.data as d


def add_filters(text):
    text = f.no_url(text)
    # text = f.username_placeholder(text)
    text = f.no_usernames(text)
    # text = f.no_emoticons(text)
    text = f.no_hash(text)
    # text = f.no_rt_tag(text)
    text = f.reduce_letter_duplicates(text)
    # text = p.remove_stopwords(text, ['not'])
    text = p.negation_attachment(text)
    return text


def test(text):
    return text


train_set_filename = '../Testing/2013-2-train-full-B.tsv'
test_set_filename = '../Testing/2013-2-test-gold-B.tsv'
my_data, my_test_data = d.set_file_names(train_set=train_set_filename, test_set=test_set_filename)

for x in my_data:
    x[3] = add_filters(x[3])

# my_data = my_data[:len(my_data)-1]
vect = CountVectorizer(preprocessor=test)
text_clf = Pipeline([
    ('vect', vect),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

print "Dataset length: %s " % len(my_data)
print("Training...")
my_clf = text_clf.fit(my_data[:, 3], my_data[:, 2])

print "# Features: %s" % len(vect.vocabulary_)

print("Done! \nClassifying test set...")
predicted = my_clf.predict(my_test_data[:, 3])

print(np.mean(predicted == my_test_data[:, 2]))

print "Accuracy: %.2f" % my_clf.score(my_data[:, 3], my_data[:, 2])
print "Accuracy: %.2f" % my_clf.score(predicted, my_test_data[:, 2])
