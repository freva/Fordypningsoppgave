from collections import defaultdict
from operator import itemgetter
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer, f1_score, precision_score
from time import time, strftime
from sklearn.pipeline import Pipeline

from storage import resource_reader as d
from storage.options import General, SubjectivityFeatures, PolarityFeatures

import utils.tokenizer as t
import utils.filters as p


def grid_search(clf, feature_pipeline, docs_train, y_train):
    t0 = time()

    pipeline = Pipeline([
        ('features', feature_pipeline),
        ('clf', clf())
    ])

    parameters = {
        'clf__kernel': ['linear'],
        'clf__C': [0.01, 0.1, 0.25, 0.4, 0.6],
        'features__word_vectorizer__ngram_range': [(1, 5)],
        'features__word_vectorizer__sublinear_tf': [True],
        'features__word_vectorizer__tokenizer': [t.tokenize],
        'features__word_vectorizer__use_idf': [True],
        'features__word_vectorizer__smooth_idf': [True],
        'features__word_vectorizer__min_df': [0.0],
        'features__word_vectorizer__max_df': [0.5],
        'features__word_vectorizer__preprocessors': [(p.html_decode, p.no_url, p.no_username, p.no_hash, p.no_emoticons, p.no_rt_tag)],
        'features__word_vectorizer__negation_scope_length': [None, -1, 4],
        'features__char_ngrams__analyzer': ['char'],
        'features__char_ngrams__ngram_range': [(3, 6)],
        'features__char_ngrams__sublinear_tf': [True],
        'features__char_ngrams__use_idf': [True],
        'features__char_ngrams__smooth_idf': [True],
        'features__char_ngrams__min_df': [0.0],
        'features__char_ngrams__max_df': [0.5],
        'features__char_ngrams__preprocessors': [(p.html_decode, p.no_url, p.no_username, p.hash_as_normal, p.no_rt_tag, p.reduce_letter_duplicates, p.limit_chars)],
        'features__char_ngrams__negation_scope_length': [None, -1, 4],
        'features__lexicon__preprocessors': [(p.html_decode, p.no_url, p.no_username, p.hash_as_normal, p.no_rt_tag, p.lower_case, p.reduce_letter_duplicates, p.limit_chars)],
        'features__lexicon__norm': [True],
        'features__pos_tagger__preprocessors': [(p.html_decode, p.no_url, p.no_username, p.no_hash, p.no_rt_tag, p.split)],
        'features__pos_tagger__norm': [True],
        'features__word_clusters__preprocessors': [(p.html_decode, p.hash_as_normal, p.no_rt_tag, p.lower_case, p.reduce_letter_duplicates)],
        'features__word_clusters__norm': [True],
        'features__punctuation__preprocessors': [(p.html_decode, p.no_url, p.no_username)],
        'features__punctuation__norm': [True],
        'features__emoticons__preprocessors': [(p.html_decode, p.no_url)],
        'features__emoticons__norm': [True],
        'features__vader__preprocessors': [(p.html_decode, p.no_url, p.no_username, p.hash_as_normal, p.no_rt_tag)],
        'features__vader__norm': [True]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid=parameters,
        scoring=make_scorer(f1_score, pos_label=None, average='binary'),
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


if __name__ == '__main__':
    train, test = d.get_data("../data/all.tsv", General.TEST_SET)
    grid_search(PolarityFeatures.CLASSIFIER['clf'], PolarityFeatures.CLASSIFIER['feature_union'], train[:, 0], train[:, 1])