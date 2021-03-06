import time

start_time = time.time()
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

# System specific
import storage.resource_reader as d
from storage.options import General, SubjectivityFeatures, PolarityFeatures

# Do import of all different methods here:
from sentiment_classifier.base import BaseMethod
from sentiment_classifier.combination import Combined

def train():
    train, test = d.get_data(General.TRAIN_SET, General.TEST_SET)

    clf = BaseMethod(train, **SubjectivityFeatures.CLASSIFIER)
    #clf = Combined(SubjectivityFeatures.CLASSIFIER, PolarityFeatures.CLASSIFIER, train)
    print "Finished training in", "%.2f" % (time.time()-start_time), "sec"
    return clf


def run_tests(clf, testsets):
    for testset in testsets:
        print testset
        test = d.read_tsv(testset)
        docs_test, y_test = test[:,0], test[:,1]
        y_pred = clf.predict(docs_test)

        out = ["%.2f" % (100*precision_score(y_test, y_pred, pos_label=None, average='macro')),
            "%.2f" % (100*recall_score(y_test, y_pred, pos_label=None, average='macro')),
            "%.2f" % (100*f1_score(y_test, y_pred, pos_label=None, average='macro')),
            "%.2f" % (100*accuracy_score(y_test, y_pred)),
            "%.2f" % (time.time()-start_time)]
        print '\t'.join(out)

        C = confusion_matrix(y_test, y_pred)
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        print 100.0*C / np.sum(C)
        print "In", "%.2f" % (time.time()-start_time), "sec"


if __name__ == "__main__":
    clf = train()

    testsets = ['../data/2013-2-test-gold-B.tsv', '../data/2014-9-test-gold-B.tsv']
    run_tests(clf, testsets)
    
