"""
    A POST Server running a sentimental analysis on a tweet stringified JSON Object.

    Takes POST requests and returns a string with the classification.
    Classification scheme: <neutral, positive, negative>
"""
import argparse
import importlib
import time
start_time = time.clock()
import sys
from collections import defaultdict

# System specific
import storage.data as d
import utils.preprocessor_methods as pr

# Do import of all different methods here:
# Remember: When adding a new method, add it to the methods/__init__.py
from models import *

d.set_file_names(train_set='../Testing/2013-2-train-full-B.tsv',
                 test_set='../Testing/2013-2-test-gold-B.tsv')
docs_test, y_test, docs_train, y_train = d.get_data()

c1_vect_options = {
    'ngram_range': (1, 1),
    'sublinear_tf': True,
    'preprocessor': pr.remove_noise,
    'use_idf': True,
    'smooth_idf': True,
    'max_df': 0.5
}

c1_default_options = {'C': 1.0}
clf = SVM(docs_train, y_train, default_options=c1_default_options, vect_options=c1_vect_options)
#clf = AFINN(docs_train, y_test, useCrossValidation=False, vect_options=c1_vect_options)
#clf = NB(docs_train, y_train, vect_options=c1_vect_options)
#clf = Boosting(docs_train, y_test)

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "test":
        importlib.import_module("test." + sys.argv[2])

    else:
        sys.argv = ["-O"]
        print sys.flags

        results, out = defaultdict(lambda: []), []

        for tweet, classification in zip(docs_test, y_test):
            results[classification].append(classification == clf.predict(tweet))

        totCor, totTot = 0, 0
        print "Class Corre Total Perce"
        for label in sorted(results.keys(), reverse=True):
            cor, tot = sum(results[label]), len(results[label])
            totCor += cor
            totTot += tot

            out.append("%.2f" % (100.0 * cor / tot))
            print label[:5], str(cor).rjust(5), str(tot).rjust(5), out[-1].rjust(5)
        out.extend(["%.2f" % (100.0 * totCor / totTot), "%.2f" % (time.clock()-start_time)])
        print "total", str(totCor).rjust(5), str(totTot).rjust(5), out[-1].rjust(5)
        print '\t'.join(out)

print "In", "%.2f" % (time.clock()-start_time), "sec"