from nltk.translate import AlignedSent, Alignment
from nltk.translate.ibm1 import IBMModel1
from nltk.translate.ibm2 import IBMModel2
from nltk import word_tokenize
from parse_data import *
import datetime


def IBM(target_listOflists_words, source_listOflists_words, flag):
	'''
	This function takes parallel corpus as input and flag variable (1 for IBM Model 1 & 2 for IBM Model 2) and outputs 
	the alignments
	'''

	model1_output_list = []

	for target_words,source_words in zip(target_listOflists_words, source_listOflists_words):
		model1_output_list.append(AlignedSent(target_words.split(),source_words.split()))

	if(flag==1):
		ibm1 = IBMModel1(model1_output_list, 50)
	elif(flag==2):
		ibm2 = IBMModel2(model1_output_list, 50)

	#print(model1_output_list)

	return model1_output_list

def tokenize(target_corpus, source_corpus):
	'''
	This helper function tokenizes the sentences into words to be used as input for NLTK implementations
	'''
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
	filename='corpus/testData.json'
	parallel_corpus=load_data_from_json(filename)
	#Creating target and source corpus
	target_corpus=[]
	source_corpus=[]
	for sent_pair in parallel_corpus:
		eng_sent=str(sent_pair['en'])
		for_sent=str(sent_pair['fr'])

		target_corpus.append(for_sent)
		source_corpus.append(eng_sent)

	#model1_raw_output = IBM([['klein', 'ist', 'das', 'haus'],['das', 'haus', 'ist', 'ja', 'gro'],['das', 'buch', 'ist', 'ja', 'klein'],['das', 'haus'],['das', 'buch'],['ein', 'buch']], [['the', 'house', 'is', 'small'], ['the', 'house', 'is', 'big'], ['the', 'book', 'is', 'small'], ['the', 'house'], ['the', 'book'], ['a', 'book']],1)

	#target_listOflists_words,source_listOflists_words = ["la maison","la fleur","la maison bleu","la fleur bleu","pomme bleu"],["the house","the flower","the blue house","the blue flower","blue apple"]
	
	#target_listOflists_words,source_listOflists_words = tokenize(target_corpus,source_corpus)

	target_listOflists_words,source_listOflists_words = target_corpus,source_corpus
	t0 = datetime.datetime.now()
	model1_raw_output = IBM(target_listOflists_words, source_listOflists_words, 1)
	t1 = datetime.datetime.now()
	model2_raw_output = IBM(target_listOflists_words, source_listOflists_words, 2)
	t2 = datetime.datetime.now()

	print(t1-t0)
	print(t2-t1)

	#print(model2_raw_output)
	
	print("##### MODEL 1 #####")
	for list in model1_raw_output:
		print(list.words)
		print(list.mots)
		print(list.alignment)
		print(" ")

	print(" ")
	print("##### MODEL 2 #####")	


	for list in model2_raw_output:
		#print(list)
		
		print(list.words)
		print(list.mots)
		print(list.alignment)
		print(" ")
		