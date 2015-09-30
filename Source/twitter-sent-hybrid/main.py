import importlib
import time

start_time = time.clock()
import sys
from collections import defaultdict

# System specific
import storage.data as d
from storage.options import Feature

# Do import of all different methods here:
# Remember: When adding a new method, add it to the methods/__init__.py
from models import *


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "test":
        importlib.import_module("test." + sys.argv[2])
    else:
        d.set_file_names(train_set  ='../Testing/2013-2-train-full-B.tsv',
                 test_set   ='../Testing/2013-2-test-gold-B.tsv')
        docs_test, y_test, docs_train, y_train = d.get_data()

        clf = SVM(docs_train, y_train, Feature.options)
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
        print "total", str(totCor).rjust(5), str(totTot).rjust(5), out[-2].rjust(5)
        print '\t'.join(out)

print "In", "%.2f" % (time.clock()-start_time), "sec"