from sklearn.feature_extraction.text import TfidfVectorizer

from transformer import *
from models import *
from sklearn.pipeline import FeatureUnion

import utils.preprocessor_methods as pr
import utils.tokenizer as t
import storage.data as d
from sklearn.svm import LinearSVC


class General:
    TRAIN_SET = '../Testing/2013-2-train-full-B.tsv'
    TEST_SET = '../Testing/2013-2-test-gold-B.tsv'
    USE_CACHE = False


class SubjectivityFeatures:
    TRANSFORMER_OPTIONS = {
        "word_vectorizer": {
            'enabled': False,
            'type': TfidfVectorizer,
            'ngram_range': (1, 4),
            'sublinear_tf': True,
            'tokenizer': t.tokenize,
            'preprocessor': pr.remove_noise,
            'use_idf': True,
            'smooth_idf': True,
            'max_df': 0.5,
            'max_features': 300000
        },

        "char_ngrams": {
            'enabled': False,
            'type': TfidfVectorizer,
            'analyzer': 'char',
            'ngram_range': (3, 5),
            'preprocessor': pr.remove_noise,
            'min_df': 1,
            'max_features': 200000
        },

        "pos_tagger": {
            'enabled': True,
            'type': POSTransformer,
        },

        "word_clusters": {
            'enabled': False,
            'type': ClusterTransformer,
            'dictionary': d.get_cluster_dict,
            'norm': True,
            # 'preprocessor': pr.remove_noise,
        },

        "allcaps": {
            'enabled': False,
            'type': AllcapsTransformer
        },

        "elongation": {
            'enabled': False,
            'type': ElongationTransformer,
            'preprocessor': pr.html_decode,
            'norm': True
        },

        "punctuation": {
            'enabled': False,
            'type': PunctuationTransformer,
            'preprocessor': pr.html_decode,
            'norm': False
        },

        "emoticons": {
            'enabled': False,
            'type': EmoticonTransformer,
            'preprocessor': pr.remove_noise,
            'norm': True
        },

        "hashtags": {
            'enabled': False,
            'type': HashtagTransformer,
            'preprocessor': pr.html_decode,
            'norm': False
        }
    }


    CLASSIFIER = {
        'clf': LinearSVC,
        'defaults': {'C': 1.0},
        'useCache': True,
        'feature_union': FeatureUnion([(name, vars.pop("type")(**TRANSFORMER_OPTIONS[name]))
                                  for name, vars in TRANSFORMER_OPTIONS.items() if vars.pop("enabled", False)])
    }


class PolarityFeatures:
    TRANSFORMER_OPTIONS = {
        "word_vectorizer": {
            'enabled': False,
            'type': TfidfVectorizer,
            'ngram_range': (1, 4),
            'sublinear_tf': True,
            'tokenizer': t.tokenize,
            'preprocessor': pr.remove_noise,
            'use_idf': True,
            'smooth_idf': True,
            'max_df': 0.5,
            'max_features': 300000
        },

        "char_ngrams": {
            'enabled': False,
            'type': TfidfVectorizer,
            'analyzer': 'char',
            'ngram_range': (3, 5),
            'preprocessor': pr.remove_noise,
            'min_df': 1,
            'max_features': 200000
        },

        "pos_tagger": {
            'enabled': True,
            'type': POSTransformer,
        },

        "word_clusters": {
            'enabled': False,
            'type': ClusterTransformer,
            'dictionary': d.get_cluster_dict,
            'norm': True,
            # 'preprocessor': pr.remove_noise,
        },

        "allcaps": {
            'enabled': False,
            'type': AllcapsTransformer
        },

        "elongation": {
            'enabled': False,
            'type': ElongationTransformer,
            'preprocessor': pr.html_decode,
            'norm': True
        },

        "punctuation": {
            'enabled': False,
            'type': PunctuationTransformer,
            'preprocessor': pr.html_decode,
            'norm': False
        },

        "emoticons": {
            'enabled': False,
            'type': EmoticonTransformer,
            'preprocessor': pr.remove_noise,
            'norm': True
        },

        "hashtags": {
            'enabled': False,
            'type': HashtagTransformer,
            'preprocessor': pr.html_decode,
            'norm': False
        }
    }


    CLASSIFIER = {
        'clf': LinearSVC,
        'defaults': {'C': 1.0},
        'useCache': True,
        'feature_union': FeatureUnion([(name, vars.pop("type")(**TRANSFORMER_OPTIONS[name]))
                                  for name, vars in TRANSFORMER_OPTIONS.items() if vars.pop("enabled", False)])
    }
