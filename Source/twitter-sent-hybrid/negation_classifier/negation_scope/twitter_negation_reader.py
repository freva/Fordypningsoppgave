# coding=utf-8
import re
from lxml.etree import parse
from storage.tweebo_cache import TweeboCacher


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

    pos_tokens = TweeboCacher.get_cached_pos_tokens()
    dependency_tweets = TweeboCacher.get_cached_dependency()

    #return [[(token, pos_tokens[i][j], dependency_tweets[i][j], is_cue, label)
    #         for j, (token, label, is_cue) in enumerate(sentence)] for i, sentence in enumerate(tweets)]

    data = [items[3].split(" ") for items in [line.split("\t") for line in open("../Testing/twitter_negation_scope.txt").read().decode("utf-8").split("\n")] if len(items) > 3]
    for i, sentence in enumerate(tweets):
        #
        if not [w[0] for w in sentence] == data[i]:
            print "1x:", i, sentence
            print len(pos_tokens[i]), pos_tokens[i]
            print len(dependency_tweets[i]), dependency_tweets[i]
            print len(data[i]), data[i]
            print [w[0] for w in sentence]
            print
        for j, (token, label, is_cue) in enumerate(sentence):
            print "2x:", j, token, pos_tokens[i][j], dependency_tweets[i][j], is_cue, label
    return []



def main():
    for tweet in parse_twitter_negation():
        print(tweet)


if __name__ == "__main__":
    main()
