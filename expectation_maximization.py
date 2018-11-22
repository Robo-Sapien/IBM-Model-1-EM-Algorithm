import numpy as np
from parse_data import *

def create_word_translation_prob_dict(parallel_corpus,foreign_name='fr'):
    '''
    This will create a dictionary mapping each french word to corresponding
    english word for keeping their probability.
    USAGE:
        INPUT:
            parallel_corpus         : the parallel corpus extracted from the
                                        json file.
        OUTPUT:
            vocab_trans_prob_dict   : the vocalbary translation probability
                                        dictionary of form:
            {
            french_word1 : {english_word1:prob,english_word2:prob,...},
            french_word2 : {english_word1:prob,english_word2:prob,...},
            ...
            ...
            }
    '''
    #Traversing the corpus to extract out the unique word
    english_word_bag={}
    foreign_word_bag={}
    init_prob_value=0.0

    for sentence in parallel_corpus:
        #Taking out the words from the sentences
        english_words=sentence['en'].split(' ')
        foreign_words=sentence[foreign_name].split(' ')

        #Hashing the words from the dict
        for word in english_words:
            english_word_bag[word]=init_prob_value
        for word in foreign_words:
            foreign_word_bag[word]=init_prob_value

    #Normalizing the probability as uniform distribution
    norm_prob=1.0/len(english_word_bag.keys())
    for key in english_word_bag.keys():
        english_word_bag[key]=norm_prob

    #Finally creating the vocab_trans_prob_dict P(ex|f1)
    for key in foreign_word_bag.keys():
        foreign_word_bag[key]=english_word_bag

    #Adding the NULL keyword in foreign to english translation dict
    foreign_word_bag['NULL']=english_word_bag

    return foreign_word_bag

def create_alignment_prob_dict(parallel_corpus,foreign_name='fr'):
    '''
    This function will keep a state table for all the probabilities
    inside of alignment within a parallel sentence.

    Alignment Dict Structure:
    {
        sentence 1: alignment frame
        sentence 2: alignment frame
        ...
        ...
    }
    '''
    alignment_dict={}
    prob_init_val=0.0

    #Iterating over all the statements and creating the alignament
    for idx,sent_pair in enumerate(parallel_corpus):
        #counting the number of word in each sentence
        eng_words=sent_pair['en'].split(' ')             #(m)
        for_words=sent_pair[foreign_name].split(' ')   #(l)

        #adding the alignment frame for each sentence
        alignment_dict[idx]=_create_alignment_prob_frame(eng_words,for_words)

    return alignment_dict

def _create_alignment_prob_frame(eng_words,for_words):
    '''
    This function will create a frame for each alignment for each
    sentence.

    Alignment Frame Strucutre (for each sentence):
    {
        e0   : {NULL:prob, f0:prob, ... fl-1:prob}
        e1   : {NULL:prob, f0:prob, ... fl-1:prob}
        .
        .
        em-1 : {NULL:prob, f0:prob, ... fl-1:prob}
    }
    '''
    word_align_prob={}
    align_prob_value=1.0/(len(for_words)+1)
    #Initializing the alignment probability
    word_align_prob['NULL']=align_prob_value
    for l in for_words:
        word_align_prob[l]=align_prob_value

    sentence_align_prob={}
    #Now adding this initilized prob for every word in source (English:m)
    for mi in eng_words:
        #For each word in the source's mi th position prob of aligning to l+1 target
        sentence_align_prob[mi]=word_align_prob

    return sentence_align_prob

def maximize_my_expectation_one_step(trans_prob,align_prob):
    '''
    This function will take the tanslation probability of each word to
    word translation along with the positional alignment probability
    and them run one step of expectation maximization by estimating the
    alignment which will in-turn change the translation probabilty.
    '''
    #Retreiving the foreign and english words
    foreign_words=trans_prob.keys()
    english_words=trans_prob[foreign_words[0]].keys()

    #Initializing the count dict
    count_trans_f2e={}
    count_f={}
    for fword in foreign_words:
        count_f[fword]=0.0
        for eword in english_words:
            count_trans_f2e[(eword,fword)]=0.0

    #Initializing the alignent count dict for each sentence
    count_align_i2j={}
    count_i={}
    for idx in align_prob.keys():
        for i in align_prob[idx].keys():#word in english (source)
            count_i[(idx,i)]=0.0
            for j in align_prob[idx][i].keys():#words in foreign
                count_align_i2j[(idx,i,j)]=0.0

    #Now we will iterate over all the examples
    for idx in align_prob.keys():
        #Now iterating over the words of source (aka english)
        for i in align_prob[idx].keys():
            #Calculating the norm for the delta
            norm=_get_expected_count_norm(idx,i,trans_prob,align_prob)
            for j in align_prob[idx][i].keys():
                #Calculating the expected count(could optimize here)
                delta=_get_expected_count(idx,i,j,trans_prob,align_prob)
                #Normalizing the delta
                delta=delta/norm

                #Now adding the expected count to the each counters
                count_trans_f2e[(i,j)]+=delta
                count_f[j]+=delta
                count_align_i2j[(idx,i,j)]+=delta
                count_i[(idx,i)]+=delta

    #Now after all count is done we will calculate the new probs
    #Calculating the probability of words
    for fword in trans_prob.keys():
        for eword in trans_prob.keys():
            trans_prob[fword][eword]=count_trans_f2e[(eword,fword)]/countf[word]

    #Calculating the new probablity of alignment
    for idx in align_prob.keys():
        for i in align_prob[idx].keys():
            for j in align_prob[idx][i].keys():
                align_prob[idx][i][j]=count_align_i2j[(idx,i,j)]/count_i[(idx,i)]

    #returning the local copy
    return trans_prob,align_prob

def _get_expected_count(idx,i,j,trans_prob,align_prob):
    '''
    This will give the expected count for a partucluar index
    for a particular ith source word to jth foreign word mapping.
    '''
    delta=align_prob[idx][i][j]*trans_prob[j][i]

def _get_expected_count_norm(idx,i,trans_prob,align_prob):
    '''
    This function will calculate the normalization for the exprected
    count probability.
    '''
    norm=0.0
    for j in align_prob[idx][i].keys():
        norm+=_get_expected_count(idx,i,j,trans_prob,align_prob)

    return norm

def expectum_maximum(trans_prob,align_prob,iteration=20):
    '''
    This function will swing the wand and maximize the expectation
    until convergence.
    '''
    for iter in range(iteration):
        _print_the_prob_dicts(trans_prob,align_prob)
        trans_prob,align_prob=maximize_my_expectation_one_step(trans_prob,
                                                        align_prob)

    return trans_prob,align_prob

def _print_the_prob_dicts(trans_prob,align_prob):
    '''
    This function will pring the probability dicts for verification
    '''
    #Printing the translation probability
    print "\n########################################"
    print "Printing the translation probability"
    for fword in trans_prob.keys():
        for eword in trans_prob[fword].keys():
            print "F:{}\t\tE:{}\t\tprob:{}".format(fword,eword,
                                        trans_prob[fword][eword])

    #printing the alignment probability
    print "\n#######################################"
    print "Printing the alignemnt proability"
    for idx in align_prob.keys():
        for i in align_prob[idx].keys():
            for j in align_prob[idx][i].keys():
                print "idx:{}\tS:{}\t\tT:{}\t\tprob:{}".format(idx,i,j,
                                                align_prob[idx][i][j])

if __name__=='__main__':
    #Loading the parallel corpus
    filename='corpus/data1.json'
    parallel_corpus=load_data_from_json(filename)

    #Initializing the word translation probability dict
    trans_prob=create_word_translation_prob_dict(parallel_corpus)
    align_prob=create_alignment_prob_dict(parallel_corpus)

    #Running the expectation maximization step
    iteration=20
    expectum_maximum(trans_prob,align_prob,iteration)
