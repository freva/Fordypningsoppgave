import fnmatch
import os
import re
from lxml.etree import parse
from storage.tweebo_cache import TweeboCacher


def find_cues(root):
    negation_cues = []
    for cue in root.iter('cue'):
        if cue.attrib['type'] == 'negation':
            negation_cues.append(cue.attrib['ID'])
    return negation_cues


def parse_file(filename):
    tree = parse(filename)
    root = tree.getroot()
    negation_cues = find_cues(root)

    sentences = []

    for sentence_tag in root.iter('SENTENCE'):
        sentence = []
        for word in sentence_tag.iter('W'):
            neg_scope_ancestors = [ancestor[0].get('SRC') in negation_cues for ancestor in word.iterancestors('xcope')]
            cleaned_text = re.sub('\u0092', "'", word.text)
            is_cue = word.getparent().tag == 'cue' and word.getparent().get('ID') in negation_cues
            if neg_scope_ancestors and sum(neg_scope_ancestors) % 2 == 1:
                sentence.append((cleaned_text, 'negated', is_cue))
            else:
                sentence.append((cleaned_text, 'affirmative', is_cue))
        sentences.append(sentence)
    return sentences


def parse_sfu():
    sentences = []
    for root, dirname, filenames in os.walk('SFU_Review_Corpus_Negation_Speculation'):
        for filename in fnmatch.filter(filenames, '*.xml'):
            sentences.extend(parse_file(os.path.join(root, filename)))
    print("Number of sentences:", len(sentences))

    TweeboCacher.cache([[token for token, label, is_cue in sentence] for sentence in sentences], True, True)
    pos_tokens = TweeboCacher.get_cached_pos_tokens()
    dependency_tweets = TweeboCacher.get_cached_dependency()

    return [[(token, pos_tokens[i][j], dependency_tweets[i][j], is_cue, label)
             for j, (token, label, is_cue) in enumerate(sentence)] for i, sentence in enumerate(sentences)]


def main():
    print(len(parse_sfu()))


if __name__ == "__main__":
    main()