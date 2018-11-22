import json
from nltk.translate.phrase_based import phrase_extraction

class PhraseBased(object):
    data = None
    phraseList = []

    """docstring for [object Object]."""
    def __init__(self, corpusFilePath):
        # with open('corpus/data2.json') as f:
        with open(corpusFilePath) as f:
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
            for phrase in phrases:
                self.phraseList.append((phrase[2], phrase[3]))

    def calculateProbabilityScore(self):
        probabilityList = []
        done = set()
        for phrase in self.phraseList:
            numerator = 0
            denominator = 0
            if tuple((phrase[1], phrase[0])) in done:
                pass
            else:
                for tup in self.phraseList:
                    if tup[0] == phrase[0]:
                        denominator += 1
                        if tup[1] == phrase[1]:
                            numerator += 1
                probability = numerator/denominator
                done.add(tuple((phrase[1], phrase[0])))
                probabilityList.append((phrase[1], phrase[0], probability))


        for i in sorted(probabilityList, key = lambda x: (-x[2],x[1])):
            print((i[1], i[0], i[2]))

if __name__ == '__main__':
    obj = PhraseBased('corpus/testData.json')
    alignments = [[(0,0), (1,1), (1,2), (1,3), (2,5), (3,6), (4,9), (5,9), (6,7), (7,7), (8,8)]]
    obj.extractPhrases(alignments)
    obj.calculateProbabilityScore()
