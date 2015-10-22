import re
import string
from storage import data
from storage.options import General
from utils import preprocessor_methods as pr

def main():
    negation_cues = ['hardly', 'lack', 'lacking', 'lacks', 'neither', 'nor', 'never', 'not', 'nobody', 'none',
                 'nothing', 'nowhere', 'no', 'without', 'aint', 'cant', 'cannot', 'darent', 'dont', 'doesnt',
                 'didnt', 'hadnt', 'hasnt', 'havent', 'havnt', 'isnt', 'mightnt', 'mustnt', 'neednt', 'oughtnt',
                 'shant', 'shouldnt', 'wasnt', 'wouldnt', ".*n't"]

    cue_pattern = re.compile('^' + '$|^'.join(negation_cues) + '$', re.IGNORECASE)

    punctuation = ['!', '?', ',', '.', '(', ')', ':']


    # naive_matcher = '('+'|'.join(negation_cues) + ')([^' + ''.join(punctuation) + ']*)(|$)'
    naive_matcher = ur"(hardly|lack|lacking|lacks|neither|nor|never|not|nobody|none|nothing|nowhere|no|without|aint|cant|cannot|darent|dont|doesnt|didnt|hadnt|hasnt|havent|havnt|isnt|mightnt|mustnt|neednt|oughtnt|shant|shouldnt|wasnt|wouldnt|.*n't)([^!?,.():]*)(|$)"
    print naive_matcher
    cue_pattern = re.compile(naive_matcher)


    training, test = data.get_data(General.TRAIN_SET, General.TEST_SET)


    for line in training[:10]:
        cleaned = pr.remove_noise(line[0])
        print cleaned
        for match in cue_pattern.findall(cleaned):
            negated_part = match[1]
            print match
        #     if any(char in punctuation for char in word):
        #         negated = False
        #     if bool(re.match(cue_pattern, word)):

            # if(bool(re.match(cue_pattern, word))):
            #     print word
    #
    # match = bool(re.match(cue_pattern, "don't"))
    # print match



if __name__ == '__main__':
    main()
