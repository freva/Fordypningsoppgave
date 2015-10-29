"""
    A collection of different filter methods for the tweets.
"""
import HTMLParser
import re
import tokenizer

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


punctuation = ['.', ',', '!', '?', ':', ';']
negation_cues = open("../data/dictionaries/negation_cues.txt", "r").read().split("\n")

word_finder = re.compile(r'(\S+)')
not_finder = re.compile(r'(^|\s)(' + '|'.join(negation_cues) + ')(\s.*?)(?=[' + ''.join(punctuation) + ']|$)', re.IGNORECASE)


def html_decode(tweet_text):
    h = HTMLParser.HTMLParser()
    return h.unescape(tweet_text).lower()


def naive_negation_attachment(tweet_text):
    return not_finder.sub((lambda m: m.group(1) + m.group(2) + word_finder.sub(r'\1_NEG', m.group(3))), tweet_text)


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

def strip_tweet(tweet_text):
    return " ".join(tweet_text.split())


def dummy(text):
    return text



punctuation = {',', '.', ':', ';', '!', '?'}
negation_cues = set(open("../data/dictionaries/negation_cues.txt", "r").read().split("\n"))
def split_into_contexts_naive(tweet):
    contexts = []
    negated = False
    for token in tokenizer.tokenize(tweet):
        token = token.lower()
        if token in negation_cues:
            negated = True
        elif token[0] in punctuation:
            negated = False
        elif negated:
            token += '_NEG'
        contexts.append(token)
    return ' '.join(contexts)


def split_into_contexts_naive2(tweet):
    contexts = []
    negated = -1
    for token in tweet.split():
        token = token.lower()
        if token in negation_cues:
            negated = 0
        elif token[0] in punctuation:
            negated = -1
        elif negated != -1 and negated < 4:
            token += '_NEG'
            negated +=1
        contexts.append(token)
    return ' '.join(contexts)