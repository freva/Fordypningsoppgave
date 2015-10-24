# coding=utf-8
import re
from lxml.etree import parse
from storage import cache

def find_cues(root):
    negation_cues = []
    for cue in root.iter('cue'):
        negation_cues.append(cue.attrib['id'])
    return negation_cues


def parse_file(filename):
    tree = parse(filename)
    root = tree.getroot()
    negation_cues = find_cues(root)

    tweets = []

    for tweet_tag in root.iter('tweet'):
        tweet = []
        for token in tweet_tag.iter('token'):
            neg_scope_ancestors = [ancestor.get('src') in negation_cues for ancestor in token.iterancestors('scope')]
            cleaned_text = re.sub('\u0092', "'", token.text)
            cleaned_text = re.sub('’', "'", cleaned_text)
            is_cue = token.getparent().tag == 'cue'
            if any(neg_scope_ancestors):
                tweet.append((cleaned_text, 'negated', is_cue))
            else:
                tweet.append((cleaned_text, 'affirmative', is_cue))
        tweets.append(tweet)
    return tweets


def parse_twitter_negation():
    tweets = parse_file('negation_classifier/negation_scope/twitter_negation_corpus.xml')

    pos_tokens = cache.load_json("neg_pos_tokens_cache")
    dependency_tweets = cache.load_json("neg_dependency_cache")

    return [[(token, pos_tokens[str(i)][j], dependency_tweets[str(i)][j], is_cue, label)
            for j, (token, label, is_cue) in enumerate(sentence)] for i, sentence in enumerate(tweets)]
