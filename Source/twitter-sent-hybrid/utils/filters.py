"""
    A collection of different filter methods for the tweets.
"""
import re

# Emoticon definitions.
NormalEyes = r'[:=8]'
HappyEyes = r'[xX]'
WinkEyes = r'[;]'
NoseArea = r'[\-o\Oc\^\*\']?'
HappyMouths = r'[dD\)\]\*\>\}]'
SadMouths = r'[c<|@L\(\[\/\{\\]'
TongueMouths = r'[pP]'

Positive_RE = re.compile('(\^_\^|' + "((" + NormalEyes + "|" + HappyEyes + "|" + WinkEyes + ")" + NoseArea + HappyMouths + ')|(?:\<3+))', re.UNICODE)
Negative_RE = re.compile(NormalEyes + NoseArea + SadMouths, re.UNICODE)

Wink_RE = re.compile(WinkEyes + NoseArea + HappyMouths, re.UNICODE)
Tongue_RE = re.compile(NormalEyes + NoseArea + TongueMouths, re.UNICODE)

Emoticon = (
    "(" + NormalEyes + "|" + HappyEyes + "|" + WinkEyes + ")" + NoseArea +
    "(" + TongueMouths + "|" + SadMouths + "|" + HappyMouths + ")"
)
Emoticon_RE = re.compile(Emoticon, re.UNICODE)

# Tag definitions
username_RE = re.compile(r'(@[a-zA-Z0-9_]{1,15})')
hashtag_RE = re.compile(r'(#[a-zA-Z]+[a-zA-Z0-9_]*)')
rt_tag_RE = re.compile(r'(^RT\s+|\s+RT\s+)')
quote_RE = re.compile(r'".*?"')
url_RE = re.compile(r'(\w+:\/\/\S+)')


def no_emoticons(tweet_text):
    tweet = re.sub(Positive_RE, "", tweet_text)
    tweet = re.sub(Negative_RE, "", tweet)
    tweet = re.sub(Emoticon_RE, "", tweet)
    return tweet


def no_username(tweet_text):
    return username_RE.sub("", tweet_text)

def username_placeholder(tweet_text):
    return username_RE.sub("||U||", tweet_text)


def no_hash(tweet_text):
    return hashtag_RE.sub("", tweet_text)

def hash_placeholder(tweet_text):
    return hashtag_RE.sub("||H||", tweet_text)


def no_rt_tag(tweet_text):
    return rt_tag_RE.sub("", tweet_text)


def no_url(tweet_text):
    return url_RE.sub("", tweet_text)

def url_placeholder(tweet_text):
    return url_RE.sub("||URL||", tweet_text)


def no_quotations(tweet_text):
    return quote_RE.sub("", tweet_text)

def quote_placeholder(tweet_text):
    return quote_RE.sub("||QUOTE||", tweet_text)


def reduce_letter_duplicates(tweet_text):
    return re.sub(r'(.)\1{3,}', r'\1\1\1', tweet_text, flags=re.IGNORECASE)


def hash_as_normal(tweet_text):
    return re.sub(r'#([a-zA-Z]+[a-zA-Z0-9_]*)', "\\1", tweet_text)

def no_punctuation(tweet_text):
    return "".join(c for c in tweet_text if c not in ('!','.',':',','))
