bing_liu_negative_path = "../Testing/lexica/BingLiu/negative-words.txt"
bing_liu_positive_path = "../Testing/lexica/BingLiu/positive-words.txt"
mpqa_lexicon_path = "../Testing/lexica/MPQA/subjclueslen1-HLTEMNLP05.tff"
nrc_emoticon_path = "../Testing/lexica/NRC-Emotion-Lexicon-v0.92/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt"

def get_automated_lexicon(filename):
    lexicon = {}
    with open(filename, 'r') as f:
        for line in f.read().decode('utf-8').split("\n"):
            newLine = line.split("\t")
            lexicon[newLine[0]] = float(newLine[1])
    return lexicon


def get_afinn_lexicon():
    lexicon = {}
    with open("../Testing/lexica/AFINN/AFINN-111.txt", 'r') as f:
        for line in f.read().decode('utf-8').split("\n"):
            newLine = line.split("\t")
            lexicon[(newLine[0])] = newLine[1]
    return lexicon

def get_bing_liu_lexicon():
    lexicon = {str(word): -3 for word in open(bing_liu_negative_path).read().decode('utf-8').split("\n")}
    lexicon.update({str(word): 3 for word in open(bing_liu_positive_path).read().decode('utf-8').split("\n")})
    return lexicon

def get_mpqa_lexicon():
    lexicon = {}
    for newLine in [line.split(" ") for line in open(mpqa_lexicon_path, 'r').read().decode('utf-8').split("\n")]:
        if newLine[5].split("=", 1)[1] == 'positive':
            lexicon[newLine[2].split("=", 1)[1]] = 2 if newLine[0][5:] == 'strongsubj' else 1
        elif newLine[5].split("=", 1)[1] == 'negative':
            lexicon[newLine[2].split("=", 1)[1]] = -2 if newLine[0][5:] == 'strongsubj'else -1
    return lexicon

def get_nrc_emotion_lexicon():
    lexicon = {}
    for newLine in [line.split("\t") for line in open(nrc_emoticon_path, 'r').read().decode('utf-8').split("\n")]:
        if newLine[2] == "1":
            if newLine[1] == 'positive':
                lexicon[newLine[0]] = 1
            elif newLine[1] == 'negative':
                lexicon[newLine[0]] = -1
    return lexicon