from nltk.translate import AlignedSent, Alignment
from nltk.translate.ibm1 import IBMModel1
from nltk.translate.ibm2 import IBMModel2
from nltk import word_tokenize


def IBM(target_listOflists_words, source_listOflists_words, flag):

	model1_output_list = []

	for target_words,source_words in zip(target_listOflists_words, source_listOflists_words):
		model1_output_list.append(AlignedSent(target_words,source_words))

	if(flag==1):
		ibm1 = IBMModel1(model1_output_list, 50)
	elif(flag==2):
		ibm2 = IBMModel2(model1_output_list, 50)

	#print(model1_output_list)

	return model1_output_list

def tokenize(target_corpus, source_corpus):
	target_listOflists_words = []
	source_listOflists_words = []
	for target_sent, source_sent in zip(target_corpus,source_corpus):
		token1 = word_tokenize(target_sent)
		token2 = word_tokenize(source_sent)
		target_listOflists_words.append(token1)
		source_listOflists_words.append(token2)

	#print(target_listOflists_words)
	#print(" ")
	#print(source_listOflists_words)

	return target_listOflists_words,source_listOflists_words

if __name__=='__main__':
	#model1_raw_output = IBM([['klein', 'ist', 'das', 'haus'],['das', 'haus', 'ist', 'ja', 'gro'],['das', 'buch', 'ist', 'ja', 'klein'],['das', 'haus'],['das', 'buch'],['ein', 'buch']], [['the', 'house', 'is', 'small'], ['the', 'house', 'is', 'big'], ['the', 'book', 'is', 'small'], ['the', 'house'], ['the', 'book'], ['a', 'book']],1)

	target_listOflists_words,source_listOflists_words = tokenize(["la maison","la fleur","la maison bleu","la fleur bleu","pomme bleu"],["the house","the flower","the blue house","the blue flower","blue apple"])
	
	model1_raw_output = IBM(target_listOflists_words, source_listOflists_words, 1)
	model2_raw_output = IBM(target_listOflists_words, source_listOflists_words, 2)

	print("MODEL 1")
	for list in model1_raw_output:
		print(list.words)
		print(list.mots)
		print(list.alignment)
		print(" ")
	print("MODEL 1")
	print(" ")
	print("MODEL 2")		
	for list in model2_raw_output:
		print(list.words)
		print(list.mots)
		print(list.alignment)
		print(" ")
	print("MODEL 2")
	