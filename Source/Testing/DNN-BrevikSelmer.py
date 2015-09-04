from json import dumps
import requests, sys

def sendPOSTRequest(text):
    urlLocal = 'http://localhost:7000/'
    urlRemote = 'http://ntnutwitsent.herokuapp.com:7000/'    
    return requests.post(urlLocal, data=text)


def sentimentFromCommandLine():
    data = ' '.join(sys.argv[1:])
    request = []
    for line in data.split(","):
        request.append(dumps({"text":  line}))

    response = sendPOSTRequest('\n'.join(request))
    print response.text
    

def testFile(filename):
    data = [i.split("\t") for i in open(filename, "r").read().split("\n")]

    request, classifications = [], []
    for tweetID, unknown, classification, text in data:
        request.append(dumps({"text":  text}))
        c = classification[1:-1]
        
        if "objective" in c:
            c = "neutral"
        classifications.append(c)

    response = sendPOSTRequest('\n'.join(request))
    response = [i for i in response.text.split("\n") if len(i) > 0]

    print sum([1 if a==b else 0 for a, b in zip(classifications, response)])


def dnnTestFile(filename):
    from sentiment import sentiment_score
    data, numCorrect = [i.split("\t") for i in open(filename, "r").read().split("\n")], 0
    i = 0
    positiveTarget, negativeTarget = 0.7, 0.3

    response, classifications = [], []
    for tweetID, unknown, classification, text in data:
        print i
        i+=1
        
        sentval = sentiment_score(text.decode("utf-8"))
        if classification == "positive":
            if sentval >= postiveTarget: numCorrect += 1
        elif classification == "negative":
            if sentval <= nevativeTarget: numCorrect += 1
        else:
            if negativeTarget < sentval < positiveTarget: numCorrect += 1
    print numCorrect
        

#testFile("test.tsv")
#dnnTestFile("test.tsv")
sentimentFromCommandLine()

