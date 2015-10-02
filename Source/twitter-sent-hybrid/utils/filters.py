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

# Username definitions
usernames = r'(@[a-zA-Z0-9_]{1,15})'


# Hashtag definitions
hashtags = r'(#[a-zA-Z]+[a-zA-Z0-9_]*)'
hashtags_filter = r'(#([a-zA-Z]+[a-zA-Z0-9_]*))'


# RT definitions
rt_tag = r'(^RT\s+|\s+RT\s+)'


# URL definitions
url = r'(\w+:\/\/\S+)'

quote = r'".*?"'


def no_emoticons(tweet_text):
    tweet = re.sub(Positive_RE, "", tweet_text)
    tweet = re.sub(Negative_RE, "", tweet)
    tweet = re.sub(Emoticon_RE, "", tweet)
    return tweet


def no_username(tweet_text):
    return re.sub(usernames, "", tweet_text)

def username_placeholder(tweet_text):
    return re.sub(usernames, "||U||", tweet_text)


def no_hash(tweet_text):
    return re.sub(hashtags, "", tweet_text)

def hash_placeholder(tweet_text):
    return re.sub(hashtags, "||H||", tweet_text)


def no_rt_tag(tweet_text):
    return re.sub(rt_tag, "", tweet_text)


def no_url(tweet_text):
    return re.sub(url, "", tweet_text)

def url_placeholder(tweet_text):
    return re.sub(url, "||URL||", tweet_text)


def no_quotations(tweet_text):
    return re.sub(quote, "", tweet_text)

def quote_placeholder(tweet_text):
    return re.sub(quote, "||QUOTE||", tweet_text)


def reduce_letter_duplicates(tweet_text):
    return re.sub(r'(.)\1{3,}', r'\1\1\1', tweet_text, flags=re.IGNORECASE)


def hash_as_normal(tweet_text):
    return re.sub(r'#([a-zA-Z]+[a-zA-Z0-9_]*)', "\\1", tweet_text)
