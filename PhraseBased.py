import json
from nltk.translate.phrase_based import phrase_extraction

class PhraseBased(object):
    data = None

    """docstring for [object Object]."""
    def __init__(self):
        # with open('corpus/data2.json') as f:
        with open('corpus/testData.json') as f:
            self.data = json.load(f)

    def extractPhrases(self,alignments):
        i = 0
        while i < len(self.data):
            sourceText = self.data[i]['en']
            targetText = self.data[i]['fr']
            # print(sourceText)
            # print(targetText)
            phrases = phrase_extraction(sourceText, targetText, alignments[i])
            i += 1
            for phrase in sorted(phrases):
                print(phrase)

# def phaseScore(arg):


if __name__ == '__main__':
    obj = PhraseBased()
    alignments = [[(0,0), (1,1), (1,2), (1,3), (2,5), (3,6), (4,9), (5,9), (6,7), (7,7), (8,8)]]
    obj.extractPhrases(alignments)
