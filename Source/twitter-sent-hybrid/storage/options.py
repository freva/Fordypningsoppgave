from sklearn.feature_extraction.text import TfidfVectorizer

from transformer import *
from models import *
from sklearn.pipeline import FeatureUnion

import utils.preprocessor_methods as pr
import utils.tokenizer as t
import storage.data as d
from sklearn.svm import LinearSVC, SVC


class General:
    TRAIN_SET = '../Testing/2013-2-train-full-B.tsv'
    TEST_SET = '../Testing/2013-2-test-gold-B.tsv'


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
            'enabled': False,
            'type': POSTransformer,
            'preprocessor': pr.remove_all,
            'norm': True
        },

        "word_clusters": {
            'enabled': False,
            'type': ClusterTransformer,
            'dictionary': d.get_cluster_dict,
            'preprocessor': pr.html_decode,
            'norm': True
        },

        "punctuation": {
            'enabled': True,
            'type': PunctuationTransformer,
            'preprocessor': pr.no_url_username,
            'norm': True
        },

        "emoticons": {
            'enabled': True,
            'type': EmoticonTransformer,
            'preprocessor': pr.no_url_username,
            'norm': True
        },

        "tags": {
            'enabled': True,
            'type': TagTransformer,
            'preprocessor': pr.html_decode,
            'norm': False
        }
    }


    CLASSIFIER = {
        'clf': SVC,
        'defaults': {'C': 1.0},
        'useCache': False,
        'feature_union': FeatureUnion([(name, vars.pop("type")(**TRANSFORMER_OPTIONS[name]))
                                  for name, vars in TRANSFORMER_OPTIONS.items() if vars.pop("enabled", False)])
    }


class PolarityFeatures:
    TRANSFORMER_OPTIONS = {
        "word_vectorizer": {
            'enabled': True,
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
            'enabled': True,
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
            'preprocessor': pr.remove_all
        },

        "word_clusters": {
            'enabled': True,
            'type': ClusterTransformer,
            'dictionary': d.get_cluster_dict,
            'norm': True,
            # 'preprocessor': pr.remove_noise,
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

        "tags": {
            'enabled': False,
            'type': TagTransformer,
            'preprocessor': pr.html_decode,
            'norm': False
        }
    }


    CLASSIFIER = {
        'clf': LinearSVC,
        'defaults': {'C': 1.0},
        'useCache': False,
        'feature_union': FeatureUnion([(name, vars.pop("type")(**TRANSFORMER_OPTIONS[name]))
                                  for name, vars in TRANSFORMER_OPTIONS.items() if vars.pop("enabled", False)])
    }
