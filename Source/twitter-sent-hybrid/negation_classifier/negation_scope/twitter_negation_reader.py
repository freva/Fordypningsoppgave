import re
from lxml.etree import parse


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
    tweets = parse_file(resources.twitter_negation_corpus)

    TweeboCacher.cache([[token for token, label, is_cue in tweet] for tweet in tweets], True, True)
    pos_tokens = TweeboCacher.get_cached_pos_tokens()
    dependency_tweets = TweeboCacher.get_cached_dependency()

    return [[(token, pos_tokens[i][j], dependency_tweets[i][j], is_cue, label)
             for j, (token, label, is_cue) in enumerate(sentence)] for i, sentence in enumerate(tweets)]


def main():
    for tweet in parse_twitter_negation():
        print(tweet)


if __name__ == "__main__":
    main()
