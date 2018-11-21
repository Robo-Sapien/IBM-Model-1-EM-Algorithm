import numpy as np

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
        num_eng=sentence_pair['en']             #(m)
        num_for=sentence_pair['foreign_name']+1 #(l+1)

        #adding the alignment frame for each sentence
        alignment_dict[idx]=_create_alignment_prob_frame(num_eng,num_for)

    return alignment_dict

def _create_alignment_prob_frame(m,l_plus_1):
    '''
    This function will create a frame for each alignment for each
    sentence.

    Alignment Frame Strucutre (for each sentence):
    {
        0   : {0:prob, 1:prob, ... l:prob}
        1   : {0:prob, 1:prob, ... l:prob}
        .
        .
        m-1 : {0:prob, 1:prob, ... l:prob}
    }
    '''
    word_align_prob={}
    align_prob_value=1.0/l_plus_1
    #Initializing the alignment probability
    for l in range(l_plus_1):
        word_align_prob[l]=align_prob_value

    sentence_align_prob={}
    #Now adding this initilized prob for every word in source (English:m)
    for mi in range(m):
        #For each word in the source's mi th position prob of aligning to l+1 target
        sentence_align_prob[mi]=word_align_prob

    return sentence_align_prob
