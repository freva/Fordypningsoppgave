from transformer import TfidfNegTransformer, LexiconTransformer, POSTransformer, ClusterTransformer, PunctuationTransformer, EmoticonTransformer
from models import *
from sklearn.pipeline import FeatureUnion

import utils.filters as f
import utils.tokenizer as t
from sklearn.svm import SVC


class General:
    TRAIN_SET = '../Testing/2013-2-train-full-B.tsv'
    TEST_SET = '../Testing/2013-2-test-gold-B.tsv'


class SubjectivityFeatures:
    TRANSFORMER_OPTIONS = {
        "word_vectorizer": {
            'enabled': True,
            'type': TfidfNegTransformer,
            'ngram_range': (1, 4),
            'tokenizer': t.tokenize,
            'preprocessors': [f.html_decode, f.no_url, f.no_username, f.no_hash, f.no_emoticons, f.no_rt_tag,
                             f.naive_negation_attachment],
            'sublinear_tf': True,
            'use_idf': True,
            'smooth_idf': True,
            'min_df': 0.0,
            'max_df': 0.5,
        },

        "char_ngrams": {
            'enabled': True,
            'type': TfidfNegTransformer,
            'analyzer': 'char',
            'ngram_range': (3, 5),
            'preprocessors': [f.html_decode, f.no_url, f.no_username, f.hash_as_normal, f.no_rt_tag,
                             f.reduce_letter_duplicates, f.quote_placeholder, f.naive_negation_attachment],
            'sublinear_tf': True,
            'use_idf': True,
            'smooth_idf': False,
            'min_df': 0.0,
            'max_df': 0.5,
        },

        "lexicon": {
            'enabled': False,
            'type': LexiconTransformer,
            'preprocessors': [f.html_decode, f.no_url, f.no_username, f.hash_as_normal, f.no_rt_tag,
                             f.reduce_letter_duplicates, f.quote_placeholder, f.naive_negation_attachment, f.strip_tweet],
            'norm': False
        },

        "pos_tagger": {
            'enabled': False,
            'type': POSTransformer,
            'norm': True
        },

        "word_clusters": {
            'enabled': False,
            'type': ClusterTransformer,
            'preprocessors': [f.html_decode],
            'norm': True
        },

        "punctuation": {
            'enabled': True,
            'type': PunctuationTransformer,
            'preprocessors': [f.html_decode, f.no_url, f.no_username],
            'norm': True
        },

        "emoticons": {
            'enabled': True,
            'type': EmoticonTransformer,
            'preprocessors': [f.html_decode, f.no_url, f.no_username],
            'norm': True
        }
    }


    CLASSIFIER = {
        'clf': SVC,
        'defaults': {
            'C': 0.5,
            'gamma': 0.001,
            'kernel': 'linear'
        },
        'useCache': False,
        'feature_union': FeatureUnion([(name, vars.pop("type")(**TRANSFORMER_OPTIONS[name]))
                                  for name, vars in TRANSFORMER_OPTIONS.items() if vars.pop("enabled", False)])
    }


class PolarityFeatures:
    TRANSFORMER_OPTIONS = {
        "word_vectorizer": {
            'enabled': True,
            'type': TfidfNegTransformer,
            'ngram_range': (1, 4),
            'sublinear_tf': True,
            'tokenizer': t.tokenize,
            'preprocessors': [f.html_decode, f.no_url, f.no_username, f.no_hash, f.no_emoticons, f.no_rt_tag,
                             f.naive_negation_attachment],
            'use_idf': True,
            'smooth_idf': True,
            'max_df': 0.5,
        },

        "char_ngrams": {
            'enabled': True,
            'type': TfidfNegTransformer,
            'analyzer': 'char',
            'ngram_range': (3, 5),
            'preprocessors': [f.html_decode, f.no_url, f.no_username, f.hash_as_normal, f.no_rt_tag,
                             f.reduce_letter_duplicates, f.quote_placeholder, f.naive_negation_attachment],
            'min_df': 1,
        },

        "lexicon": {
            'enabled': True,
            'type': LexiconTransformer,
            'preprocessors': [f.html_decode, f.no_url, f.no_username, f.hash_as_normal, f.no_rt_tag,
                             f.reduce_letter_duplicates, f.quote_placeholder, f.naive_negation_attachment],
        },

        "pos_tagger": {
            'enabled': True,
            'type': POSTransformer,
            'norm': True,
        },

        "word_clusters": {
            'enabled': True,
            'type': ClusterTransformer,
            'preprocessors': [f.html_decode],
            'norm': True,
        },

        "punctuation": {
            'enabled': False,
            'type': PunctuationTransformer,
            'preprocessors': [f.html_decode, f.no_url, f.no_username],
            'norm': True
        },

        "emoticons": {
            'enabled': False,
            'type': EmoticonTransformer,
            'preprocessors': [f.html_decode, f.no_url, f.no_username],
            'norm': True
        }
    }


    CLASSIFIER = {
        'clf': SVC,
        'defaults': {
            'C': 0.5,
            'gamma': 0.001,
            'kernel': 'linear'
        },
        'useCache': False,
        'feature_union': FeatureUnion([(name, vars.pop("type")(**TRANSFORMER_OPTIONS[name]))
                                  for name, vars in TRANSFORMER_OPTIONS.items() if vars.pop("enabled", False)])
    }
