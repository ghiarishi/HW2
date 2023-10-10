from multiprocessing import Pool
import numpy as np
import time
from tagger_utils import *
from math import log
import math
import csv
from collections import defaultdict 
from collections import Counter
import copy
from itertools import permutations


""" Contains the part of speech tagger class. """


def evaluate(data, model):
    """Evaluates the POS model on some sentences and gold tags.

    This model can compute a few different accuracies:
        - whole-sentence accuracy
        - per-token accuracy
        - compare the probabilities computed by different styles of decoding

    You might want to refactor this into several different evaluation functions,
    or you can use it as is. 
    
    As per the write-up, you may find it faster to use multiprocessing (code included). 
    
    """
    processes = 4
    sentences = data[0]
    tags = data[1]
    n = len(sentences)
    k = n//processes
    n_tokens = sum([len(d) for d in sentences])
    unk_n_tokens = sum([1 for s in sentences for w in s if w not in model.word2idx.keys()])
    predictions = {i:None for i in range(n)}
    probabilities = {i:None for i in range(n)}
         
    start = time.time()
    pool = Pool(processes=processes)
    res = []

    for i in range(0, n, k):
        res.append(pool.apply_async(infer_sentences, [model, sentences[i:i+k], i]))
    

    ans = [r.get(timeout=None) for r in res]
    predictions = dict()
    for a in ans:
        predictions.update(a)
    print(f"Inference Runtime: {(time.time()-start)/60} minutes.")
    
    start = time.time()
    pool = Pool(processes=processes)
    res = []
    for i in range(0, n, k):
        res.append(pool.apply_async(compute_prob, [model, sentences[i:i+k], tags[i:i+k], i]))
    ans = [r.get(timeout=None) for r in res]
    probabilities = dict()
    for a in ans:
        probabilities.update(a)
    
    print(f"Probability Estimation Runtime: {(time.time()-start)/60} minutes.")

    token_acc = sum([1 for i in range(n) for j in range(len(sentences[i])) if tags[i][j] == predictions[i][j]]) / n_tokens
    unk_token_acc = sum([1 for i in range(n) for j in range(len(sentences[i])) if tags[i][j] == predictions[i][j] and sentences[i][j] not in model.word2idx.keys()]) / unk_n_tokens
    whole_sent_acc = 0
    num_whole_sent = 0
    for k in range(n):
        sent = sentences[k]
        eos_idxes = indices(sent, '.')
        start_idx = 1
        end_idx = eos_idxes[0]
        for i in range(1, len(eos_idxes)):
            whole_sent_acc += 1 if tags[k][start_idx:end_idx] == predictions[k][start_idx:end_idx] else 0
            num_whole_sent += 1
            start_idx = end_idx+1
            end_idx = eos_idxes[i]
    print("Whole sent acc: {}".format(whole_sent_acc/num_whole_sent))
    print("Mean Probabilities: {}".format(sum(probabilities.values())/n))
    print("Token acc: {}".format(token_acc))
    print("Unk token acc: {}".format(unk_token_acc))
    
    confusion_matrix(pos_tagger.tag2idx, pos_tagger.idx2tag, predictions.values(), tags, 'cm.png')

    return whole_sent_acc/num_whole_sent, token_acc, sum(probabilities.values())/n


class POSTagger():
    def __init__(self):
        """Initializes the tagger model parameters and anything else necessary. """
        
        
        self.unigramsCount = {} # count of each unique unigram 
        self.bigramsCount = {} # count of each unique bigram
        self.trigramsCount = {} # count of each unique trigram
        self.emissionsCount = {} # count of each unique unigram

        # INPUT HERE
        self.k = 0.1 # add-k smoothing hyperparameter

        self.smoothing = True # false is witten (for bigrams) and linear interpolation for trigrams), true is add k
        self.model = 3 # 1 for greedy, 2 for beam, 3 for viterbi
        self.kgram = 3 # 2 for bigrams, 3 for trigrams   
        self.beam_k = 3 # k parameter as input to beam search
    
    def get_unigrams(self):
        """
        Computes unigrams. 
        Tip. Map each tag to an integer and store the unigrams in a numpy array. 

        Actually think of this function as a way to get the transition probability 
        Which is basically the probability of a word being a noun or some other tag. 
        So actually need to count the frequency of a certain tag, and divide by the total no of tags. 
        """
        ## TODO
        unigram = np.zeros(len(self.all_tags))
        for tag in self.tag2idx: 
            unigram[self.tag2idx[tag]] = self.unigramsCount[tag]/self.N

    def get_bigrams(self):        
        """
        Computes bigrams. 
        Tip. Map each tag to an integer and store the bigrams in a numpy array
             such that bigrams[index[tag1], index[tag2]] = Prob(tag2|tag1). 

        
        So basically this gives you the transition probability of tag2/tag1
        """
        ## TODO

        tag2idx = self.tag2idx

        # initialize the count of each bigram to 0
        self.bigramsCount = {(tag2idx[tag1], tag2idx[tag2]): 0 for tag1 in tag2idx for tag2 in tag2idx}

        # increment the count of a bigram by 1 each time it appears
        for sentence in self.data[1]: 
            for i in range(len(sentence)-1): 
                tag1 = sentence[i]
                tag2 = sentence[i+1]
                self.bigramsCount[(self.tag2idx[tag1], self.tag2idx[tag2])] += 1

        # calculate the transition probability 
        self.bigrams = np.zeros((len(self.all_tags), len(self.all_tags)))
        
        # smoothing
        for key, count in self.bigramsCount.items(): 
            tag1 = self.idx2tag[key[0]]
            tag2 = self.idx2tag[key[1]]


            if self.smoothing: # add-k smoothing
                denominator = self.unigramsCount[tag1] + self.k*self.V
                self.bigrams[key[0],key[1]] = (count + self.k)/denominator
            
            else: # witten-bell smoothing
                denominator = (len(self.T[tag1]) + self.unigramsCount[tag1])
                if count == 0: # case where bigram never occurs
                    self.bigrams[key[0],key[1]] = len(self.T[tag1])/denominator
                else: 
                    self.bigrams[key[0],key[1]] = (count)/denominator     

    def get_trigrams(self):
        """
        Computes trigrams. 
        Tip. Similar logic to unigrams and bigrams. Store in numpy array. 
        """
        tag2idx = self.tag2idx
        all_tags_len = len(self.all_tags)

        # initialize the count of each trigram to 0
        trigramsCount = {(tag2idx[tag1], tag2idx[tag2], tag2idx[tag3]): 0 for tag1 in tag2idx for tag2 in tag2idx for tag3 in tag2idx}
        
        # increment the count of each trigram by 1 when it appears
        for sentence in self.data[1]: 
            for i in range(1, len(sentence)):
                # handle the case where we need two start tags (for the first word after 'O')
                tag1 = 'O' if i == 1 else sentence[i-2]
                tag2 = 'O' if i == 1 else sentence[i-1]
                tag3 = sentence[i]
                
                trigram_key = (tag2idx[tag1], tag2idx[tag2], tag2idx[tag3])
                trigramsCount[trigram_key] += 1

        
        trigrams = np.zeros((all_tags_len, all_tags_len, all_tags_len))
        bigramsCount = self.bigramsCount
        unigramsCount = self.unigramsCount
        idx2tag = self.idx2tag
        k = self.k
        V = self.V
        N = self.N

        for trigram, count in trigramsCount.items(): 
            tag1_idx, tag2_idx, tag3_idx = trigram
            tag1, tag2, tag3 = idx2tag[tag1_idx], idx2tag[tag2_idx], idx2tag[tag3_idx]

            if self.smoothing: # add-k smoothing
                denominator = bigramsCount[tag1_idx, tag2_idx] + k * V
                trigrams[tag1_idx, tag2_idx, tag3_idx] = (count + k) / denominator
            
            else: # linear interpolation 
                # hyperparameter values for unigrams, bigrams, and trigrams
                lambda1, lambda2, lambda3 = 1/3, 1/3, 1/3  

                # Fetch the required counts/probabilities
                bigram_count = bigramsCount.get((tag2_idx, tag3_idx), 0)
                trigram_prob = 0 if bigram_count == 0 else count / bigram_count
                bigram_prob = bigram_count / unigramsCount[tag2]
                unigram_prob = unigramsCount[tag3] / N

                interpolated_prob = lambda3 * trigram_prob + lambda2 * bigram_prob + lambda1 * unigram_prob
                trigrams[tag1_idx, tag2_idx, tag3_idx] = interpolated_prob

        self.trigrams = trigrams

    def get_emissions(self):
        """
        Computes emission probabilities. 
        Tip. Map each tag to an integer and each word in the vocabulary to an integer. 
             Then create a numpy array such that lexical[index(tag), index(word)] = Prob(word|tag) 

        Probability of word given a tag, to find this you need to count instances of the word, given a tag
        """
        ## TODO
        data = self.data
        tag2idx = self.tag2idx
        idx2tag = self.idx2tag

        # iterate through every word
        for i in range(len(data[0])): 
            for j in range(len(data[0][i])):
                if (data[0][i][j], tag2idx[data[1][i][j]]) not in self.emissionsCount: 
                    self.emissionsCount[(data[0][i][j], tag2idx[data[1][i][j]])] = 1
                else: 
                    self.emissionsCount[(data[0][i][j], tag2idx[data[1][i][j]])] += 1

        self.emissions = np.zeros((len(self.all_words),len(self.all_tags)))

        # calculate the emission probabilities
        for key,value in self.emissionsCount.items():
            denom = self.unigramsCount[idx2tag[key[1]]]
            self.emissions[self.word2idx[key[0]], key[1]] = value/denom


    def train(self, data):
        """Trains the model by computing transition and emission probabilities.

        You should also experiment:
            - smoothing.
            - N-gram models with varying N.
        
        """
        self.data = data  # data[0] has all the words in the data set

        self.all_tags = list(set([t for tag in data[1] for t in tag]))  # This is the list of all the PoS tags in the dataset. 

        self.all_words = list(set([word for sentence in data[0] for word in sentence]))  # This is the list of all the PoS tags in the dataset. 
        
        self.word2idx = {self.all_words[i]:i for i in range(len(self.all_words))}  # This is basically a dictionary of Tag : id 
        
        self.idx2word = {v:k for k,v in self.word2idx.items()}    # And this basically is a dictionary of id: Tag

        self.tag2idx = {self.all_tags[i]:i for i in range(len(self.all_tags))}  # This is basically a dictionary of Tag : id 
        
        self.idx2tag = {v:k for k,v in self.tag2idx.items()}    # And this basically is a dictionary of id: Tag

        

        ## TODO
        # number of unique tags appearing after each tag
        self.T = {key: set() for key in self.all_tags} # used for witten bell smoothing
        
        for sentence in data[1]:
            for i in range(len(sentence)-1):
                self.T[sentence[i]].add(sentence[i+1])

        # count of each tag 
        for tag in self.all_tags: 
            self.unigramsCount[tag] = 0
        for sentence in data[1]: 
            for tag in sentence: 
                self.unigramsCount[tag] += 1

        self.N = sum(self.unigramsCount.values())
        
        self.V = len(set(self.word2idx.keys()))
        
        self.get_unigrams()

        self.get_bigrams()

        self.get_trigrams()

        self.get_emissions()

        # map each bigram to an index, and vice-versa
        self.bigram2idx = {tup:idx for idx,tup in enumerate(self.bigramsCount.keys())} 

        self.idx2bigram = {idx:tup for tup,idx in self.bigram2idx.items()} 

        # Build the suffix mapping to deal with unknown words
        self.suffixes = defaultdict(Counter)
        for sentence, tag_seq in zip(data[0], data[1]):
            for word, tag in zip(sentence, tag_seq):
                # Here, we take the last 3 characters as the suffix
                self.suffixes[word[-3:]].update([tag])

        # Build the prefix mapping to deal with unknown words
        self.prefixes = defaultdict(Counter)
        for sentence, tag_seq in zip(data[0], data[1]):
            for word, tag in zip(sentence, tag_seq):
                # Here, we take the first 2 characters as the prefix
                self.prefixes[word[2:]].update([tag])
                
        # Choose the most common tag for each suffix
        self.suffix_to_tag = {suffix: tags.most_common(1)[0][0] for suffix, tags in self.suffixes.items()}
        self.prefix_to_tag = {prefix: tags.most_common(1)[0][0] for prefix, tags in self.prefixes.items()}        

    def sequence_probability(self, sequence, tags):
        """Computes the probability of a tagged sequence given the emission/transition
        probabilities.
        """
        ## TODO
       
        prob = 1
        for i in range(1, len(sequence)):
            # probability of word given the tag
            if sequence[i] in self.word2idx:
                q = self.bigrams[self.tag2idx[tags[i-1]], self.tag2idx[tags[i]]]
                e = self.emissions[self.word2idx[sequence[i]], self.tag2idx[tags[i]]]
                prob *= q*e
            else: 
                continue
                # FILL THIS IN 
                # Unknown word prob = 1

        return prob
    
    def inference(self, sequence):
        """Tags a sequence with part of speech tags.

        You should implement different kinds of inference (suggested as separate
        methods):

            - greedy decoding
            - decoding with beam search
            - viterbi
        """
        # probably won't use this function. 
        ## TODO

        # run the correct model based on the given self.model value
        if self.model == 1: 
            seq = self.greedy(sequence)
        elif self.model == 2: 
            seq = self.beam(sequence, self.beam_k)
        elif self.model == 3: 
            seq = self.viterbi(sequence)
        return seq

    def greedy (self, sequence):
        """ Tags a sequence with PoS tags

        Implements Greedy decoding"""
        ## TODO 
        
        # bigram case
        if self.kgram == 2: 
            
            # set the starting tag to 'O'
            prev = 'O'
            tagSeq = ['O']

            for word in sequence[1:]: # iterate through every word after the start word
        
                maxi = 0
                maxTag = ''

                if word in self.word2idx: # known word

                    # iterate through all tags, and identify the previous tag that leads to the maximum bigram transition probability
                    for tag2 in self.all_tags: 
                        q = self.bigrams[self.tag2idx[prev], self.tag2idx[tag2]] # calculate bigram transition probability
                        e = self.emissions[self.word2idx[word], self.tag2idx[tag2]]
                        if q*e > maxi: 
                            maxTag = tag2
                            maxi = q*e
                    prev = maxTag
                    tagSeq.append(maxTag)

                else: # deal with unknown words 
                    cur_tag = self.suffix_to_tag.get(word[-3:], None)  # default to none if suffix not in mapping
                    if(cur_tag == None):
                        if(word[0].isupper()): # classify the word as a Proper Noun ig the first letter is upper case
                            cur_tag = "NNP"
                        else:
                            cur_tag = self.prefix_to_tag.get(word[2:], None)  # default to none if suffix not in mapping
                            if(cur_tag == None):
                                cur_tag = "NN" # if none of the above conditions true, default to noun 
                    tagSeq.append(cur_tag)
            return tagSeq
        
        elif self.kgram == 3: # trigram case

            # set both prev tags to 'O' to get the first transition probability
            prev1 = 'O' # prev1 is the tag right before the current tag
            prev2 = 'O' # prev2 is the tag 2 tags back from the current tag

            tagSeq = ['O']
            
            for word in sequence[1:]:  # iterate through all words in the sequence after the first one
                maxi = 0
                maxTag = ''

                if word in self.word2idx: # known case
                    for tag2 in self.all_tags: 
                        q = self.trigrams[self.tag2idx[prev2], self.tag2idx[prev1], self.tag2idx[tag2]] # calculate trigram probability
                        e = self.emissions[self.word2idx[word], self.tag2idx[tag2]]
                        if q*e > maxi: 
                            maxTag = tag2
                            maxi = q*e
                    prev2 = prev1
                    prev1 = maxTag
                    tagSeq.append(maxTag)
                
                else: # unknown case 
                    cur_tag = self.suffix_to_tag.get(word[-3:], None)  # default to none if suffix not in mapping
                    if(cur_tag == None):
                        if(word[0].isupper()): # classify the word as a Proper Noun ig the first letter is upper case
                            cur_tag = "NNP"
                        else:
                            cur_tag = self.prefix_to_tag.get(word[2:], None)  # default to none if suffix not in mapping
                            if(cur_tag == None):
                                cur_tag = "NN" # if none of the above conditions true, default to noun
                    tagSeq.append(cur_tag)
            return tagSeq

    def beam(self, sequence, k):
        tag2idx = self.tag2idx
        word2idx = self.word2idx
        all_tags = self.all_tags
        bigrams = self.bigrams
        trigrams = self.trigrams
        emissions = self.emissions
        unigramsCount = self.unigramsCount
        N = self.N
    
        if self.kgram == 2: # bigram case

            # store the overall most probable k sequences and their probabilities
            top_k_seq = [['O'] for _ in range(k)]
            top_k_prob = [0] * k

            for word in sequence[1:-1]:

                if word in word2idx: # known case
                    word_idx = word2idx[word]

                    # dict to store probabilities, and their respective sequences, and parent sequence indices
                    cur_k_seq = {}
                    
                    # set to store the current top k sequences that are in the above dict 
                    cur_k_seq_set = set()

                    # iterate through the current top k sequences
                    for i, parent in enumerate(top_k_seq):

                        parent_prob = top_k_prob[i]
                        last_tag_idx = tag2idx[parent[-1]]

                        for tag in all_tags:
                            tag_idx = tag2idx[tag]
                            q = bigrams[last_tag_idx, tag_idx]
                            e = emissions[word_idx, tag_idx]
                            
                            if e == 0: continue
                            
                            # use log prob to avoid zeroing out of probabilities 
                            prob = log(q) + log(e) + parent_prob
                            new_seq = parent + [tag]

                            # the set ensures we do not add in a duplicate sequence
                            if tuple(new_seq) not in cur_k_seq_set:

                                # add in a sequence without checking if we have under k sequences
                                if len(cur_k_seq_set) < k:
                                    cur_k_seq_set.add(tuple(new_seq))
                                    cur_k_seq[prob] = new_seq

                                else: # replace mininum probability sequence with higher probability sequence
                                    minProb = min(cur_k_seq.keys())
                                    if prob > minProb:
                                        remove_seq = cur_k_seq.pop(minProb)
                                        cur_k_seq_set.remove(tuple(remove_seq))
                                        cur_k_seq_set.add(tuple(new_seq))
                                        cur_k_seq[prob] = new_seq

                    top_k_prob = sorted(cur_k_seq.keys(), reverse=True)[:k]
                    top_k_seq = [cur_k_seq[p] for p in top_k_prob]

                else:
                    for i, seq in enumerate(top_k_seq):
                        cur_tag = self.suffix_to_tag.get(word[-3:], None)  # default to none if suffix not in mapping
                        if(cur_tag == None):

                            if(word[0].isupper()): # classify the word as a Proper Noun ig the first letter is upper case
                                cur_tag = "NNP"
                            else:
                                cur_tag = self.prefix_to_tag.get(word[2:], None)  # default to none if suffix not in mapping
                                if(cur_tag == None):
                                    cur_tag = "NN" # if none of the above conditions true, default to noun
                        seq.append(cur_tag)
                        
                        last_tag_idx = tag2idx[seq[-1]]
                        cur_tag_idx = tag2idx[cur_tag]
                        q = bigrams[last_tag_idx, cur_tag_idx]
                        e = unigramsCount[cur_tag] / N
                        top_k_prob[i] += log(q) + log(e)

            # return the highest probability sequence from the top k identified
            sol = top_k_seq[0]
            sol.append('.')
            return sol

        elif self.kgram == 3:

            # store the overall most probable k sequences and their probabilities
            top_k_seq = [['O', 'O'] for _ in range(k)]
            top_k_prob = [0] * k

            for word in sequence[1:-1]:

                if word in word2idx: # known case
                    word_idx = word2idx[word]

                    # dict to store probabilities, and their respective sequences, and parent sequence indices
                    cur_k_seq = {}
                    
                    # set to store the current top k sequences that are in the above dict 
                    cur_k_seq_set = set()

                    # iterate through the current top k sequences
                    for i, parent in enumerate(top_k_seq):
                        parent_prob = top_k_prob[i]
                        last_two_tags_idx = (tag2idx[parent[-2]], tag2idx[parent[-1]])

                        for tag in all_tags:
                            tag_idx = tag2idx[tag]
                            q = trigrams[last_two_tags_idx + (tag_idx,)]
                            e = emissions[word_idx, tag_idx]

                            if e == 0: continue
                            
                            # use log prob to avoid zeroing out of probabilities 
                            prob = log(q) + log(e) + parent_prob
                            new_seq = parent + [tag]

                            # the set ensures we do not add in a duplicate sequence
                            if tuple(new_seq) not in cur_k_seq_set:
                                
                                # add in a sequence without checking if we have under k sequences
                                if len(cur_k_seq_set) < k:
                                    cur_k_seq_set.add(tuple(new_seq))
                                    cur_k_seq[prob] = new_seq

                                else: # replace mininum probability sequence with higher probability sequence
                                    minProb = min(cur_k_seq.keys())
                                    if prob > minProb:
                                        remove_seq = cur_k_seq.pop(minProb)
                                        cur_k_seq_set.remove(tuple(remove_seq))
                                        cur_k_seq_set.add(tuple(new_seq))
                                        cur_k_seq[prob] = new_seq

                    top_k_prob = sorted(cur_k_seq.keys(), reverse=True)[:k]
                    top_k_seq = [cur_k_seq[p] for p in top_k_prob]

                else:
                    for i, seq in enumerate(top_k_seq):
                        cur_tag = self.suffix_to_tag.get(word[-3:], None)  # default to none if suffix not in mapping
                        if(cur_tag == None):
                            if(word[0].isupper()): # classify the word as a Proper Noun ig the first letter is upper case
                                cur_tag = "NNP"
                            else:
                                cur_tag = self.prefix_to_tag.get(word[2:], None)  # default to none if suffix not in mapping
                                if(cur_tag == None):
                                    cur_tag = "NN" # if none of the above conditions true, default to noun
                        seq.append(cur_tag)
                        
                        last_two_tags_idx = (tag2idx[seq[-2]], tag2idx[seq[-1]])
                        cur_tag_idx = tag2idx[cur_tag]
                        q = trigrams[last_two_tags_idx + (cur_tag_idx,)]
                        e = unigramsCount[cur_tag] / N
                        top_k_prob[i] += log(q) + log(e)

            # return the highest probability sequence from the top k identified
            sol = top_k_seq[0]
            sol.append('.')
            sol.pop(0)
            return sol  

    def viterbi (self, sequence):
        """ Tags a sequence with PoS tags

        Implements viterbi decoding"""

        # TODO

        len_bigramsCount = len(self.bigramsCount.keys())
        log = math.log
        tag2idx = self.tag2idx
        bigram2idx = self.bigram2idx
        idx2tag = self.idx2tag
        idx2bigram = self.idx2bigram
        bigrams = self.bigrams
        trigrams = self.trigrams
        emissions = self.emissions
        word2idx = self.word2idx
        N = self.N

        if self.kgram == 2: 
            pi = np.full((len(sequence), len(self.all_tags)), -math.inf)

            pi[0] = 0

            bp = np.zeros((len(sequence), len(self.all_tags)))

            for i in range(1,len(sequence)): 

                word = sequence[i]

                if word in self.word2idx: # if word is known
                    
                    for j, cur_tag in enumerate(self.all_tags):
                        e = emissions[word2idx[word], tag2idx[cur_tag]]
                        if e > 0: 
                            e = log(e)
                            for k, prev in enumerate(self.all_tags):
                                q = log(bigrams[k, tag2idx[cur_tag]])
                                prod = q + e + pi[i-1, k]
                                
                                if prod > pi[i, j]:
                                    pi[i, j] = prod
                                    bp[i, j] = k

                else: # if word is unknown
                    
                    cur_tag = self.suffix_to_tag.get(word[-3:], None)  # default to none if suffix not in mapping

                    if(cur_tag == None):

                        if(word[0].isupper()): # classify the word as a Proper Noun ig the first letter is upper case
                            cur_tag = "NNP"
                        else:
                            cur_tag = self.prefix_to_tag.get(word[2:], None)  # default to none if suffix not in mapping
                            if(cur_tag == None):
                                cur_tag = "NN" # if none of the above conditions true, default to noun

                    j = self.tag2idx[cur_tag]
                    e = self.unigramsCount[cur_tag]/self.N  #1 / len(self.word2idx)
                    maxPtag = -math.inf  # Initialize with negative infinity
                    
                    for k in range(len(self.all_tags)):  # Loop over all possible previous tags

                        q = self.bigrams[k, j]
                        
                        # If q and e are both non-zero, compute the Viterbi score
                        if q*e > 0:
                            prob = log(q) + log(e) + pi[i-1, k]
                            if prob > maxPtag:
                                maxPtag = prob
                                bp[i, j] = k
                    
                    pi[i, j] = maxPtag 

            # Reconstruct the max sequence:            
            seq = ['O'] + [''] * (len(sequence) - 2) + [self.idx2tag[np.argmax(pi[-1])]]
            for i in range(len(sequence)-2, 0, -1):
                max_idx = np.argmax(pi[i+1])
                seq[i] = self.idx2tag[int(bp[i+1, max_idx])]
            return seq
                
        elif self.kgram == 3:
                    
            pi = np.full((len(sequence), len_bigramsCount), -math.inf)
            pi[0] = 0

            bp = np.zeros((len(sequence), len_bigramsCount))
            
            # iterate through all the words, starting from the one after the start tag
            for i in range(1,len(sequence)): 

                word = sequence[i]

                if word in self.word2idx: # if word is known
                    
                    for j, cur_tag in enumerate(self.all_tags):
                        e = emissions[word2idx[word], tag2idx[cur_tag]]

                        if e > 0: 
                            e = log(e)
                            for index, k in enumerate(self.bigramsCount.keys()):

                                if i == 1:
                                    q = log(bigrams[0, tag2idx[cur_tag]])
                                else:
                                    q = log(trigrams[k[0], k[1], tag2idx[cur_tag]])
                                prod = q + e + pi[i-1, index]
                                new_bigram = (k[1], tag2idx[cur_tag])
                                bigram_idx = bigram2idx[new_bigram]
                                if prod > pi[i, bigram_idx]:
                                    pi[i, bigram_idx] = prod
                                    bp[i, bigram_idx] = k[1]

        
                else: # if word is unknown
                    cur_tag = self.suffix_to_tag.get(word[-3:], None)  # default to none if suffix not in mapping

                    if(cur_tag == None):

                        if(word[0].isupper()): # classify the word as a Proper Noun ig the first letter is upper case
                            cur_tag = "NNP"
                        else:
                            cur_tag = self.prefix_to_tag.get(word[2:], None)  # default to none if suffix not in mapping
                            if(cur_tag == None):
                                cur_tag = "NN" # if none of the above conditions true, default to noun
                    
                    j = self.tag2idx[cur_tag]

                    e = self.unigramsCount[cur_tag]/self.N  #1 / len(self.word2idx)

                    maxPtag = -math.inf  # Initialize with negative infinity
                    
                    index = 0
                    for k in self.bigramsCount.keys():  # Loop over all possible previous bigrams
                        q = self.trigrams[k[0],k[1], j]

                        new_bigram = (k[1], self.tag2idx[cur_tag])
                        
                        bigram_idx = self.bigram2idx[new_bigram]

                        # If q and e are both non-zero, compute the Viterbi score
                        if e > 0:
                            prob = log(q) + log(e) + pi[i-1, index]
                            
                            if prob > maxPtag:
                                maxPtag = prob
                                bp[i, bigram_idx] = k[1]
                                pi[i, bigram_idx] = maxPtag 
                        index+=1 

            # Reconstruct the max sequence: 
            seq = ['O'] + [''] * (len(sequence) - 2) + [idx2tag[idx2bigram[np.argmax(pi[-1])][1]]]
            for i in range(len(sequence)-2, 0, -1):
                max_idx = np.argmax(pi[i+1])
                seq[i] = idx2tag[int(bp[i+1, max_idx])]
            return seq

if __name__ == "__main__":
    pos_tagger = POSTagger()

    train_data = load_data("data/train_x.csv", "data/train_y.csv")
    dev_data = load_data("data/dev_x.csv", "data/dev_y.csv")
    test_data = load_data("data/test_x.csv")

    pos_tagger.train(train_data)

    # Experiment with your decoder using greedy decoding, beam search, viterbi...

    # Here you can also implement experiments that compare different styles of decoding,
    # smoothing, n-grams, etc.

    evaluate(dev_data, pos_tagger)



    # Predict tags for the test set
    test_predictions = []
    for sentence in test_data:
        test_predictions.extend(pos_tagger.inference(sentence))
    
    # Write them to a file to update the leaderboard
    # TODO
    
    with open("test_y.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "tag"])  # write the headers first
        for index, item in enumerate(test_predictions):
            writer.writerow([index, item])