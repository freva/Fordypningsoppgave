"""
    A collection of different functions for preprocessing tweets.
    Used together with filtering
"""

import re, HTMLParser
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.snowball import SnowballStemmer
from spellcheck import correct



def _negation_repl(matchobj):
    """
      Internal helper method for #negation_attachment.
    """
    if matchobj.group(2):
        if matchobj.group(1):
            return matchobj.group(1) + "-not not-" + matchobj.group(2)
        else:
            return "not-" + matchobj.group(2)

    if not matchobj.group(1) and not matchobj.group(2):
        return "not"

    return matchobj.group(1) + "-not"


def negation_attachment(tweet_text):
    """
      Attaches the negation word "not" to prefixed and suffixed words.

      Examples:
      This is not perfect at all! => This is-not not-perfect at all!
      I am not!! => I am-not!!
      I'm not!! => I'm-not!!
      I am not short => I am-not not-short.
    """
    tweet_text = tweet_text.replace("n't", " not")
    return re.sub(r'([\S]+)?(?:\s+)?(?:not)(?:\s+)?([a-zA-Z][\S]+)?', _negation_repl, tweet_text, flags=re.IGNORECASE)


def remove_stopwords(tweet_text, exceptionList=[]):
    """
      Used to remove stop words from a tweet.
      exceptionList is a list of words that are the exception to the rule.
      So even if "not" is a stopword, remove_stopwords("not", ["not"]) == "not"
    """
    tweet_text = tweet_text.lower()
    word_list = wordpunct_tokenize(tweet_text)
    filtered_words = [w for w in word_list if not w in stopwords.words('english') or w in exceptionList]
    return " ".join(filtered_words)


def html_decode(tweet_text):
    h = HTMLParser.HTMLParser()
    return h.unescape(tweet_text).lower()


def stem_sentence(tweet_text):
    pattern = re.compile(ur'[a-z]+')
    stemmer = SnowballStemmer("english")

    spellcheck = [correct(word.encode('ascii', 'ignore')) if pattern.match(word) else word for word in wordpunct_tokenize(tweet_text.lower())]

    return ' '.join(stemmer.stem(word) for word in spellcheck)


