"""
    A collection of different functions for preprocessing tweets.
    Used together with filtering
"""

import re, HTMLParser


punctuation = ['.', ',', '!', '?', '(', ')']
negation_cues = open("../Testing/dictionaries/negation_cues.txt", "r").read().split("\n")

word_finder = re.compile(r'(\S+)')
not_finder = re.compile(r'(^|\s)(' + '|'.join(negation_cues) + ')(\s.*?)(?=[' + ''.join(punctuation) + ']|$)', re.IGNORECASE)


def html_decode(tweet_text):
    h = HTMLParser.HTMLParser()
    return h.unescape(tweet_text).lower()


def naive_negation_attachment(tweet_text):
    return not_finder.sub((lambda m: m.group(1) + m.group(2) + word_finder.sub(r'\1_NEG', m.group(3))), tweet_text)


