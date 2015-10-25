import time

start_time = time.clock()
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

# System specific
import storage.resource_reader as d
from storage.options import General, SubjectivityFeatures, PolarityFeatures

# Do import of all different methods here:
from sentiment_classifier.base import BaseMethod
from sentiment_classifier.combination import Combined


if __name__ == "__main__":
    train, test = d.get_data(General.TRAIN_SET, General.TEST_SET)
    docs_test, y_test = test[:,0], test[:,1]

    clf = BaseMethod(train, **SubjectivityFeatures.CLASSIFIER)
    #clf = Combined(SubjectivityFeatures.CLASSIFIER, PolarityFeatures.CLASSIFIER, train)
    print "Finished training in", "%.2f" % (time.clock()-start_time), "sec"
    y_pred = clf.predict(docs_test)

    """out = ["%.2f" % (100*precision_score(y_test, y_pred, pos_label="neutral", average='binary')),
        "%.2f" % (100*recall_score(y_test, y_pred, pos_label="neutral", average='binary')),
        "%.2f" % (100*f1_score(y_test, y_pred, pos_label="neutral", average='binary')),
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
