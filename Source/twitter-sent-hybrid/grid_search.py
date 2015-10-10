from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer, f1_score, precision_score
from time import time
from sklearn.pipeline import Pipeline

from storage import data as d
from storage.options import General, SubjectivityFeatures
import utils.preprocessor_methods as pr


def grid_search(clf, feature_pipeline, docs_train, y_train):
    t0 = time()

    pipeline = Pipeline([
        ('features', feature_pipeline),
        ('clf', clf())
    ])

    parameters = {
        'clf__kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
        'clf__C': (0.5, 1.0),
        'clf__gamma': (0.001, 0.1, 0.02),
        'features__punctuation__preprocessor': (pr.html_decode, pr.no_url_username),
        'features__punctuation__norm': (True, False),
        'features__emoticons__preprocessor': [pr.html_decode],
        'features__emoticons__norm': (True, False),
        'features__tags__preprocessor': [pr.html_decode],
        'features__tags__norm': (True, False)
    }

    grid = GridSearchCV(
        pipeline,
        param_grid=parameters,
        scoring=make_scorer(f1_score, pos_label="neutral", average='binary'),
        cv=StratifiedKFold(y_train, 2),#, shuffle=True), #IDI maskinen sin scikit-learn stotter ikke shuffle (gammel versjon)
        n_jobs=-1,
        verbose=10
    )

    grid.fit(docs_train, y_train)

    f = open("results.txt", "w")
    f.write("Performed SVM grid search in %0.3fs\n" % (time() - t0))
    f.write("Best grid search CV score: {:0.3f}\n".format(grid.best_score_))
    f.write("Best parameters set:\n")

    best_parameters = grid.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        f.write("\t%s: %r\n" % (param_name, best_parameters[param_name]))

    f.write("\nParam scores:\n")
    for s in grid.grid_scores_:
        f.write(str.format("{0:.3f}", s.cv_validation_scores.mean()*100) + "\t" +
                str.format("{0:.3f}", s.cv_validation_scores.std()*100) + "\t" + str(s.parameters) + "\n")
    f.close()

    return grid.best_estimator_

if __name__ == '__main__':
    train, test = d.get_data(General.TRAIN_SET, General.TEST_SET)
    grid_search(SubjectivityFeatures.CLASSIFIER['clf'], SubjectivityFeatures.CLASSIFIER['feature_union'], train[:,0], train[:,1])