import importlib
import time

start_time = time.clock()
import sys
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

# System specific
import storage.data as d
from storage.options import General, SubjectivityFeatures

# Do import of all different methods here:
# Remember: When adding a new method, add it to the methods/__init__.py
from models.base import BaseMethod


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "test":
        importlib.import_module("test." + sys.argv[2])
    else:
        docs_test, y_test, docs_train, y_train = d.get_data(General.TRAIN_SET, General.TEST_SET)

        #clf = BaseMethod(docs_train, y_train, **SubjectivityFeatures.CLASSIFIER)
        clf =
        print "Finished training in", "%.2f" % (time.clock()-start_time), "sec"
        y_pred = clf.predict(docs_test)

        """out = ["%.2f" % (100*precision_score(y_test, y_pred, average='macro')),
            "%.2f" % (100*recall_score(y_test, y_pred, average='macro')),
            "%.2f" % (100*f1_score(y_test, y_pred, average='macro')),
            "%.2f" % (100*accuracy_score(y_test, y_pred)),
            "%.2f" % (time.clock()-start_time)]
        print '\t'.join(out)"""

        out = ["%.2f" % (100*precision_score(y_test, y_pred, pos_label=None, average='macro')),
            "%.2f" % (100*recall_score(y_test, y_pred, pos_label=None, average='macro')),
            "%.2f" % (100*f1_score(y_test, y_pred, pos_label=None, average='macro')),
            "%.2f" % (100*accuracy_score(y_test, y_pred)),
            "%.2f" % (time.clock()-start_time)]
        print '\t'.join(out)

        C = confusion_matrix(y_test, y_pred)
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        print 100.0*C / np.sum(C)

print "In", "%.2f" % (time.clock()-start_time), "sec"
