import sys
import math
from operator import itemgetter
import numpy as np
import json

def predict(text):
    lexicon = np.load('lexicon.npy').item()
    probability = np.load('probability.npy').item()
    categories = np.load('categories.npy')
    def classification(tweet):
        def splitting(something):
            return something.split()
        def stripping(something):
            return something.strip('#":?".;\/')
        def lowering(something):
            return something.lower()
        output = {}
        out = output
        cat = categories
        
        for i in cat:
            out[i] = 0
        
        tokenization = splitting(tweet)
        for eachtoken in tokenization:
            eachtoken = stripping(eachtoken)
            eachtoken = lowering(eachtoken)
            if eachtoken in lexicon:
                for i in cat:
                    prob = probability[i]
                    if prob[eachtoken] == 0:
                        """print("%s %s" % (i, eachtoken))"""
                    out[i] += math.log(prob[eachtoken])
  
        out = list(out.items())
        out.sort(key=itemgetter(1), reverse = True)
        return out
    prediction = classification(text)
    
    def softmax(w):
        e = np.exp(w)
        dist = e / np.sum(e)
        return dist
        
    emotions = [emotion for (emotion, score) in prediction]
    scores = [score for (emotion, score) in prediction]
    probs = softmax(scores)
    result = []
    for i in xrange(len(emotions)):
      result.append((emotions[i], probs[i]))
    return json.dumps(result)

if __name__ == "__main__":
  text = sys.argv[1]
  print predict(text)