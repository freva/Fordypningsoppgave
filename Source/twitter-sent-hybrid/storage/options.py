from sentiment_classifier.transformer import TfidfNegTransformer, LexiconTransformer, POSTransformer, \
    ClusterTransformer, PunctuationTransformer, EmoticonTransformer, VaderTransformer
from sklearn.pipeline import FeatureUnion

import utils.filters as f
import utils.tokenizer as t
from sklearn.svm import SVC


class General:
    TRAIN_SET = '../data/2013-2-train-full-B.tsv'
    TEST_SET = '../data/2013-2-test-gold-B.tsv'


class SubjectivityFeatures:
    TRANSFORMER_OPTIONS = {
        "word_vectorizer": {
            'enabled': True,
            'type': TfidfNegTransformer,
            'ngram_range': (1, 5),
            'tokenizer': t.tokenize,
            'preprocessors': [f.html_decode, f.no_url, f.no_username, f.no_hash, f.no_emoticons, f.no_rt_tag],
            'sublinear_tf': True,
            'use_idf': True,
            'smooth_idf': True,
            'min_df': 0.0,
            'max_df': 0.5,
            'negation_scope_length': 4
        },

        "char_ngrams": {
            'enabled': True,
            'type': TfidfNegTransformer,
            'analyzer': 'char',
            'ngram_range': (3, 6),
            'preprocessors': [f.html_decode, f.no_url, f.no_username, f.hash_as_normal, f.no_rt_tag,
                              f.reduce_letter_duplicates, f.limit_chars],
            'sublinear_tf': True,
            'use_idf': True,
            'smooth_idf': False,
            'min_df': 0.0,
            'max_df': 0.5,
            'negation_scope_length': None
        },

        "lexicon": {
            'enabled': True,
            'type': LexiconTransformer,
            'preprocessors': [f.html_decode, f.no_url, f.no_username, f.hash_as_normal, f.no_rt_tag, f.lower_case,
                              f.reduce_letter_duplicates, f.limit_chars, f.naive_negation_attachment],
            'norm': True
        },

        "pos_tagger": {
            'enabled': True,
            'type': POSTransformer,
            'preprocessors': [f.html_decode, f.no_url, f.no_username, f.no_hash, f.no_rt_tag, f.split],
            'norm': True
        },

        "word_clusters": {
            'enabled': True,
            'type': ClusterTransformer,
            'preprocessors': [f.html_decode, f.hash_as_normal, f.no_rt_tag, f.lower_case, f.reduce_letter_duplicates],
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
            'preprocessors': [f.html_decode, f.no_url],
            'norm': True
        },

        "vader": {
            'enabled': True,
            'type': VaderTransformer,
            'preprocessors': [f.html_decode, f.no_url, f.no_username, f.hash_as_normal, f.no_rt_tag],
            'norm': True
        }
    }

    CLASSIFIER = {
        'clf': SVC,
        'defaults': {
            'kernel': 'linear',
            'C': 0.1,
        },
        'useCache': True,
        'feature_union': FeatureUnion([(name, vars.pop("type")(**TRANSFORMER_OPTIONS[name]))
                                       for name, vars in TRANSFORMER_OPTIONS.items() if vars.pop("enabled", False)])
    }


class PolarityFeatures:
    TRANSFORMER_OPTIONS = {
        "word_vectorizer": {
            'enabled': True,
            'type': TfidfNegTransformer,
            'ngram_range': (1, 5),
            'tokenizer': t.tokenize,
            'preprocessors': [f.html_decode, f.no_url, f.no_username, f.no_hash, f.no_emoticons, f.no_rt_tag],
            'sublinear_tf': True,
            'use_idf': True,
            'smooth_idf': True,
            'min_df': 0.0,
            'max_df': 0.5,
            'negation_scope_length': -1
        },

        "char_ngrams": {
            'enabled': True,
            'type': TfidfNegTransformer,
            'analyzer': 'char',
            'ngram_range': (2, 5),
            'preprocessors': [f.html_decode, f.no_url, f.no_username, f.hash_as_normal, f.no_rt_tag,
                              f.reduce_letter_duplicates, f.limit_chars],
            'sublinear_tf': True,
            'use_idf': True,
            'smooth_idf': False,
            'min_df': 0.0,
            'max_df': 0.5,
            'negation_scope_length': None
        },

        "lexicon": {
            'enabled': True,
            'type': LexiconTransformer,
            'preprocessors': [f.html_decode, f.no_url, f.no_username, f.hash_as_normal, f.no_rt_tag, f.lower_case,
                              f.reduce_letter_duplicates, f.limit_chars],
            'norm': True
        },

        "pos_tagger": {
            'enabled': True,
            'type': POSTransformer,
            'preprocessors': [f.html_decode, f.no_url, f.no_username, f.no_hash, f.no_rt_tag, f.split],
            'norm': True
        },

        "word_clusters": {
            'enabled': True,
            'type': ClusterTransformer,
            'preprocessors': [f.html_decode, f.hash_as_normal, f.no_rt_tag, f.lower_case, f.reduce_letter_duplicates],
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
            'preprocessors': [f.html_decode, f.no_url],
            'norm': True
        },

        "vader": {
            'enabled': True,
            'type': VaderTransformer,
            'preprocessors': [f.html_decode, f.no_url, f.no_username, f.hash_as_normal, f.no_rt_tag],
            'norm': True
        }
    }

    CLASSIFIER = {
        'clf': SVC,
        'defaults': {
            'kernel': 'linear',
            'C': 0.01,
        },
        'useCache': True,
        'feature_union': FeatureUnion([(name, vars.pop("type")(**TRANSFORMER_OPTIONS[name]))
                                       for name, vars in TRANSFORMER_OPTIONS.items() if vars.pop("enabled", False)])
    }
