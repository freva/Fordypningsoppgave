import HTMLParser
import re
from subprocess import Popen, PIPE, STDOUT

import io
import pickle

tweets_to_cache = ['../2013-2-train-full-B.tsv', '../2013-2-test-gold-B.tsv']


def read_tsv(filename):
    return [line.split("\t") for line in open(filename).read().decode("windows-1252").split("\n") if len(line) > 0]


def preprocessor(tweet_text):
    h = HTMLParser.HTMLParser()
    return h.unescape(tweet_text).lower()


def main():
    raw_tweets, raw_pos = [], []
    f = io.open("tweets.txt", "w", encoding="utf-8")
    for filename in tweets_to_cache:
        tweets = read_tsv(filename)
        for tweet in tweets:
            raw_tweets.append(tweet[3])
            tweet = tweet[3].encode("ascii", "ignore")
            tweet = unicode(preprocessor(tweet).replace("\n", "").replace("\r", ""))
            f.write(tweet + "\n")
    f.close()

    p = Popen(['java', '-Xmx1024m', '-jar', 'twitie_tag.jar', 'models/gate-EN-twitter.model', 'tweets.txt'],
              stdout=PIPE, stderr=STDOUT)

    pos_tags_RE = re.compile(ur'_([A-Z]+)\s')
    for line in p.stdout:
        if "_" not in line: continue
        raw_pos.append(pos_tags_RE.findall(line))

    full_path = '../../twitter-sent-hybrid/pickles/pos_tags.pkl'
    output = open(full_path, 'wb')
    pickle.dump(dict(zip(raw_tweets, raw_pos)), output, protocol=2)
    output.close()

main()