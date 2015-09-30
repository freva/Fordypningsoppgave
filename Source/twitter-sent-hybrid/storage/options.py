import storage.data as d
import utils.preprocessor_methods as pr

class Feature:
    WORD_VECTORIZER = "word_vectorizer"
    CHAR_NGRAMS = "char_ngrams"
    WORD_CLUSTERS = "word_clusters"
    SVM_DEFAULT_OPTIONS = "default_options"

    options = {
        WORD_VECTORIZER: {
           'ngram_range': (1, 4),
            'sublinear_tf': True,
            'preprocessor': pr.remove_noise,
            'use_idf': True,
            'smooth_idf': True,
            'max_df': 0.5
        },

        CHAR_NGRAMS: {
            'ngram_range': (3, 5),
            'min_df': 1
        },

        WORD_CLUSTERS: {
            'dictionary': d.get_cluster_dict(),
            'norm': True
        },

        SVM_DEFAULT_OPTIONS: {
            'C': 1.0
        }
    }