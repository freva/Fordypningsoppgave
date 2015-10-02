import storage.data as d
import utils.preprocessor_methods as pr

from sklearn.feature_extraction.text import TfidfVectorizer

from transformer import ClusterTransformer
from transformer import AllcapsTransformer
from transformer import WordCounter
from transformer import ElongationTransformer
from transformer import PunctuationTransformer
from sklearn.pipeline import FeatureUnion


class Feature:
    WORD_VECTORIZER = "word_vectorizer"
    CHAR_NGRAMS = "char_ngrams"
    WORD_CLUSTERS = "word_clusters"
    ALL_CAPS = "allcaps"
    ELONGATION = "elongation"
    PUNCTUATION = "punctuation"
    SVM_DEFAULT_OPTIONS = "default_options"
    WORD_COUNT = "word_count"

    options = {
        WORD_VECTORIZER: {
            'enabled': True,
            'type': TfidfVectorizer,
            'ngram_range': (1, 4),
            'sublinear_tf': True,
            'preprocessor': pr.remove_noise,
            'use_idf': True,
            'smooth_idf': True,
            'max_df': 0.5,
            'max_features': 300000
        },

        CHAR_NGRAMS: {
            'enabled': True,
            'type': TfidfVectorizer,
            'ngram_range': (3, 5),
            'preprocessor': pr.remove_noise,
            'min_df': 1,
            'max_features': 200000
        },

        WORD_CLUSTERS: {
            'enabled': True,
            'type': ClusterTransformer,
            'dictionary': d.get_cluster_dict(),
            'norm': True,
            # 'preprocessor': pr.remove_noise,
        },

        ALL_CAPS: {
            'enabled': True,
            'type': AllcapsTransformer
        },

        ELONGATION: {
            'enabled': True,
            'type': ElongationTransformer
        },

        PUNCTUATION: {
            'enabled': True,
            'type': PunctuationTransformer
        },

        WORD_COUNT: {
            'enabled': True,
            'type': WordCounter
        },

        SVM_DEFAULT_OPTIONS: {
            'C': 1.0
        }
    }

    feature_union = FeatureUnion([(name, vars.pop("type")(**options[name])) for name, vars in options.items() if vars.pop("enabled", False)])