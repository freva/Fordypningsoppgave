from collections import defaultdict
from operator import itemgetter
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer, f1_score, precision_score
from time import time, strftime
from sklearn.pipeline import Pipeline

from storage import data as d
from storage.options import General, SubjectivityFeatures

import utils.preprocessor_methods as pr
import utils.tokenizer as t


def grid_search(clf, feature_pipeline, docs_train, y_train):
    t0 = time()

    pipeline = Pipeline([
        ('features', feature_pipeline),
        ('clf', clf())
    ])

    parameters = {
        'clf__kernel': ('linear', 'rbf'),
        'clf__C': (0.5, 0.75, 1.0, 1.25),
        'clf__gamma': (0.001, 0.1, 0.3, 0.5),
        'features__word_vectorizer__ngram_range': [(1, 4)], #(1,4) slight better than (1,3) and (1,5), then (1,2) and (1,1) is worst
        'features__word_vectorizer__sublinear_tf': [True],  #Virtually identical
        'features__word_vectorizer__tokenizer': [t.tokenize],
        'features__word_vectorizer__use_idf': [True],       #False appears to be slightly better on average, but is not in optimal
        'features__word_vectorizer__smooth_idf': [True],    #Virtually identical
        'features__word_vectorizer__max_df': [0.5],#(0.25, 0.5, 0.75, 1), #0.5 significantly better than 1.0
        'features__word_vectorizer__max_features': (None, 100000, 200000, 300000),
        'features__word_vectorizer__preprocessor': [pr.remove_all],
        'features__char_ngrams__analyzer': ['char'],
        'features__char_ngrams__ngram_range': [(2, 5), (3, 5), (3, 6), (3, 7)],
        'features__char_ngrams__sublinear_tf': (True, False),
        'features__char_ngrams__use_idf': (True, False),
        'features__char_ngrams__smooth_idf': (True, False),
        'features__char_ngrams__min_df': (0.25, 0.5, 0.75, 1),
        'features__char_ngrams__max_features': (None, 100000, 200000, 300000),
        'features__char_ngrams__preprocessor': (pr.remove_all, pr.remove_noise),
        #'features__word_clusters__dictionary': [d.get_cluster_dict],
        #'features__word_clusters__preprocessor': [pr.html_decode],      #Dictionary contains hashtags, emotes and even usernames. Dont think any further filtering is possible
        #'features__word_clusters__norm': (True, False),
        'features__punctuation__preprocessor': [pr.no_url_username],    #Remove url and username works better than just html decode
        'features__punctuation__norm': [True],              #No preference
        'features__emoticons__preprocessor': [pr.html_decode],  #Technically could run with no_url_username //overlap with tags
        'features__emoticons__norm': [True]                 #0.01% difference in favor of True
    }

    grid = GridSearchCV(
        pipeline,
        param_grid=parameters,
        scoring=make_scorer(precision_score, pos_label="neutral", average='binary'),
        cv=StratifiedKFold(y_train, 5, shuffle=True),
        n_jobs=-1,
        verbose=10
    )

    grid.fit(docs_train, y_train)

    f = open("results.txt", "w")
    f.write("Generated at: " + str(strftime("%Y-%m-%d %H:%M")) + "\n")
    f.write("Performed SVM grid search in %0.3fs\n" % (time() - t0))
    f.write("Best grid search CV score: {:0.3f}\n".format(100*grid.best_score_))
    f.write("Best parameters set:\n")

    best_parameters = grid.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        f.write("\t%s: %r\n" % (param_name, best_parameters[param_name]))

    cat_score = defaultdict(list)
    for s in grid.grid_scores_:
        for cat in s.parameters.keys():
            cat_score[cat + "=" + str(s.parameters[cat])].append(s.cv_validation_scores.mean()*100)

    f.write("\nEfficient params:\n")
    cat_score = {key: sum(val)/len(val) for key, val in cat_score.items()}
    for k, v in sorted(cat_score.items(), key=itemgetter(0), reverse=True):
        f.write("\t" + k + "\t" + str(v) + "\n")

    f.write("\nParam scores:\n")
    for s in grid.grid_scores_:
        f.write(str.format("{0:.3f}", s.cv_validation_scores.mean()*100) + "\t" +
                str.format("{0:.3f}", s.cv_validation_scores.std()*100) + "\t" + str(s.parameters) + "\n")

    f.close()

    return grid.best_estimator_


if __name__ == '__main__':
    train, test = d.get_data(General.TRAIN_SET, General.TEST_SET)
    grid_search(SubjectivityFeatures.CLASSIFIER['clf'], SubjectivityFeatures.CLASSIFIER['feature_union'], train[:,0], train[:,1])