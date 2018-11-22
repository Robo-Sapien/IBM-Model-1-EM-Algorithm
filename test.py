from nltk.translate import AlignedSent, Alignment
from nltk.translate.ibm1 import IBMModel1
from nltk.translate.ibm2 import IBMModel2

bitext = []

bitext.append(AlignedSent(['klein', 'ist', 'das', 'haus'], ['the', 'house', 'is', 'small']))
bitext.append(AlignedSent(['das', 'haus', 'ist', 'ja', 'gro'], ['the', 'house', 'is', 'big']))
bitext.append(AlignedSent(['das', 'buch', 'ist', 'ja', 'klein'], ['the', 'book', 'is', 'small']))
bitext.append(AlignedSent(['das', 'haus'], ['the', 'house']))
bitext.append(AlignedSent(['das', 'buch'], ['the', 'book']))
bitext.append(AlignedSent(['ein', 'buch'], ['a', 'book']))

#print(bitext)

ibm1 = IBMModel1(bitext, 50)
print(bitext)

'''
test_sent = bitext[2]
print(test_sent.words)
print(test_sent.mots)
print(test_sent.alignment)
'''

#for row in bitext:
#	print(row)

#print(bitext)