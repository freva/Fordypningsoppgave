import os

RESOURCES_DIR = os.path.dirname(__file__)


lexica = {'s140_u': os.path.join(RESOURCES_DIR,'lexica/Sentiment140/S140-AFFLEX-NEGLEX-unigrams.txt'),
          's140_b':  os.path.join(RESOURCES_DIR,'lexica/Sentiment140/S140-AFFLEX-NEGLEX-bigrams.txt'),
          'hs_u': os.path.join(RESOURCES_DIR,'lexica/HashtagSentiment/HS-AFFLEX-NEGLEX-unigrams.txt'),
          'hs_b': os.path.join(RESOURCES_DIR,'lexica/HashtagSentiment/HS-AFFLEX-NEGLEX-bigrams.txt'),
          'nrc_e':  os.path.join(RESOURCES_DIR,'lexica/NRC-Emotion-Lexicon-v0.92/''NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt'),
          'bing_p':  os.path.join(RESOURCES_DIR,'lexica/BingLiu/positive-words.txt'),
          'bing_n':  os.path.join(RESOURCES_DIR,'lexica/BingLiu/negative-words.txt'),
          'mpqa':  os.path.join(RESOURCES_DIR,'lexica/MPQA/subjclueslen1-HLTEMNLP05.tff')
          }
