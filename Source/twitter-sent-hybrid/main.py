import importlib
import time

start_time = time.clock()
import sys
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

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
        print "Finished training in", "%.2f" % (time.clock()-start_time), "sec"
        y_pred = clf.predict(docs_test)

        out = ["%.2f" % (100*precision_score(y_test, y_pred, average='macro')),
            "%.2f" % (100*recall_score(y_test, y_pred, average='macro')),
            "%.2f" % (100*f1_score(y_test, y_pred, average='macro')),
            "%.2f" % (100*accuracy_score(y_test, y_pred)),
            "%.2f" % (time.clock()-start_time)]
        print '\t'.join(out)


print "In", "%.2f" % (time.clock()-start_time), "sec"