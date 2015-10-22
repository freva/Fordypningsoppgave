import re
import lxml.etree as et
from utils.twokenizer import tokenize


def wrap_words(element):
    if len(element):
        for child in element.iterchildren():
            wrap_words(child)
            if child.tail:
                tokens = tokenize(child.tail)
                child.tail = ""
                for token in reversed(tokens):
                    word = et.Element('word')
                    word.text = token
                    element.insert(element.index(child) + 1, word)
    if element.text:
        tokens = tokenize(element.text)
        element.text = ""
        for token in reversed(tokens):
            word = et.Element('word')
            word.text = token
            element.insert(0, word)


def fix_bioscope(file):
    parser = et.XMLParser(remove_blank_text=True)
    root = et.parse(file, parser).getroot()
    for sentence in root.iter('sentence'):
        wrap_words(sentence)
    tree = et.ElementTree(root)
    with open(resources.bioscope_corpus, 'wb') as f:
        tree.write(f)



def find_cues(root):
    negation_cues = []
    for cue in root.iter('cue'):
        if cue.attrib['type'] == 'negation':
            negation_cues.append(cue.attrib['ref'])
    return negation_cues


def parse_file(filename):
    tree = et.parse(filename)
    root = tree.getroot()
    negation_cues = find_cues(root)

    sentences = []

    for sentence_tag in root.iter('sentence'):
        sentence = []
        for word in sentence_tag.iter('word'):
            neg_scope_ancestors = [ancestor.get('id') in negation_cues for ancestor in word.iterancestors('xcope')]
            cleaned_text = re.sub('\u0092', "'", word.text)
            is_cue = (word.getparent().tag == 'cue') and (word.getparent().get('ref') in negation_cues)
            if is_cue:
                if sum(neg_scope_ancestors) > 1:
                    sentence.append((cleaned_text, 'negated', True))
                else:
                    sentence.append((cleaned_text, 'affirmative', True))
            else:
                if any(neg_scope_ancestors):
                    sentence.append((cleaned_text, 'negated', False))
                else:
                    sentence.append((cleaned_text, 'affirmative', False))
        sentences.append(sentence)
    return sentences


def parse_bioscope():
    sentences = parse_file(resources.bioscope_corpus)
    print("Number of sentences:", len(sentences))

    TweeboCacher.cache([[token for token, label, is_cue in sentence] for sentence in sentences], True, True)
    pos_tokens = TweeboCacher.get_cached_pos_tokens()
    dependency_tweets = TweeboCacher.get_cached_dependency()

    return [[(token, pos_tokens[i][j], dependency_tweets[i][j], is_cue, label)
             for j, (token, label, is_cue) in enumerate(sentence)] for i, sentence in enumerate(sentences)]


def main():
    a = parse_bioscope()
    print("hello")
    print("wtf")


if __name__ == "__main__":
    main()
