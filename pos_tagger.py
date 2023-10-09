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
    # print(len(predictions[5]))
    # print(len(tags[5]))
    # print(len(sentences[5]))
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
    
    # print(len(pos_tagger.tag2idx), len(pos_tagger.idx2tag), len(predictions.values()), len(tags))
    confusion_matrix(pos_tagger.tag2idx, pos_tagger.idx2tag, predictions.values(), tags, 'cm.png')

    return whole_sent_acc/num_whole_sent, token_acc, sum(probabilities.values())/n


class POSTagger():
    def __init__(self):
        """Initializes the tagger model parameters and anything else necessary. """
        
        self.tagCounts = {}
        self.bigramsCount = {}
        self.trigramsCount = {}
        self.emissionsCount = {}
        self.k = 0.1
        self.smoothing = False # false is witten, true is add k
        self.unknowns = False # false is suffix, true is nouns 
        self.beam_k = 3
        self.model = 2
        self.kgram = 3
    
    
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
            unigram[self.tag2idx[tag]] = self.tagCounts[tag]/self.N

    def get_bigrams(self):        
        """
        Computes bigrams. 
        Tip. Map each tag to an integer and store the bigrams in a numpy array
             such that bigrams[index[tag1], index[tag2]] = Prob(tag2|tag1). 

        
        So basically this gives you the transition probability of tag2/tag1
        """
        ## TODO

        for tag1 in self.tag2idx: 
            for tag2 in self.tag2idx: 
                self.bigramsCount[(self.tag2idx[tag1], self.tag2idx[tag2])] = 0

        # count the self.bigramsCount
        for sentence in self.data[1]: 
            for i in range(len(sentence)-1): 
                tag1 = sentence[i]
                tag2 = sentence[i+1]
                self.bigramsCount[(self.tag2idx[tag1], self.tag2idx[tag2])] += 1

        # count total bigrams, and use it to divide

        self.bigrams = np.zeros((len(self.all_tags), len(self.all_tags)))
        
        # Implementing add-k smoothing, witten-bell smoothing


        for key, count in self.bigramsCount.items(): 
            tag1 = self.idx2tag[key[0]]
            tag2 = self.idx2tag[key[1]]

            if self.smoothing: 
                denominator = self.tagCounts[tag1] + self.k*self.V
                self.bigrams[key[0],key[1]] = (count + self.k)/denominator
            
            else: 
                denominator = (len(self.T[tag1]) + self.tagCounts[tag1])

                if count == 0: # bigram never occurs
                    self.bigrams[key[0],key[1]] = len(self.T[tag1])/denominator
                else: 
                    self.bigrams[key[0],key[1]] = (count)/denominator     
            
    def get_trigrams(self):
        """
        Computes trigrams. 
        Tip. Similar logic to unigrams and bigrams. Store in numpy array. 
        """
        ## TODO

        # PREPEND ONE EXTRA START : Lecture 4 slide 19

        # Maybe, we will make a 3D np array, with all the possibilities and initialize them to 0. 

        for tag1 in self.tag2idx: 
            for tag2 in self.tag2idx: 
                for tag3 in self.tag2idx:
                    self.trigramsCount[(self.tag2idx[tag1], self.tag2idx[tag2],self.tag2idx[tag3])] = 0

        
        # count the self.trigramsCount
        for sentence in self.data[1]: 
            for i in range(1,len(sentence)):
                if i ==1:
                    tag1 = 'O'
                    tag2 = 'O'
                else:  
                    tag1 = sentence[i-2]
                    tag2 = sentence[i-1]
                tag3 = sentence[i]
                self.trigramsCount[(self.tag2idx[tag1], self.tag2idx[tag2], self.tag2idx[tag3])] += 1

        # Implementing add-k smoothing 
        
        self.trigrams = np.zeros((len(self.all_tags), len(self.all_tags),len(self.all_tags)))

        for trigram, count in self.trigramsCount.items(): 
            tag1 = self.idx2tag[trigram[0]]
            tag2 = self.idx2tag[trigram[1]]
            tag3 = self.idx2tag[trigram[2]]

            if self.smoothing: 
                denominator = self.bigramsCount[self.tag2idx[tag1],self.tag2idx[tag2]] + self.k*self.V 
                self.trigrams[trigram[0],trigram[1],trigram[2]] = (count + self.k)/denominator
            else: 
                if count == 0: # bigram never occurs
                    z = 0
                    for key in self.trigramsCount.keys(): 
                        if key[0] == tag1 and key[1] == tag2: 
                            for tag_last in self.all_tags: 
                                if self.trigramsCount[(tag1, tag2, tag_last)] == 0: 
                                    z += 1
                    denominator = (len(self.T2[(tag1, tag2)]) + self.bigramsCount[tag1, tag2])*z
                    self.trigrams[tag1, tag2, tag3] = self.bigramsCount[tag1, tag2]/denominator
                else: 
                    denominator = len(self.T2[(tag1, tag2)]) + self.bigramsCount[tag1, tag2]
                    self.trigrams[tag1, tag2, tag3] = count/denominator     

    def get_emissions(self):
        """
        Computes emission probabilities. 
        Tip. Map each tag to an integer and each word in the vocabulary to an integer. 
             Then create a numpy array such that lexical[index(tag), index(word)] = Prob(word|tag) 

        Probability of word given a tag, to find this you need to count instances of the word, given a tag
        """
        ## TODO
        
        for i in range(len(self.data[0])): 
            for j in range(len(self.data[0][i])):
                if (self.data[0][i][j], self.tag2idx[self.data[1][i][j]]) not in self.emissionsCount: 
                    self.emissionsCount[(self.data[0][i][j], self.tag2idx[self.data[1][i][j]])] = 1
                else: 
                    self.emissionsCount[(self.data[0][i][j], self.tag2idx[self.data[1][i][j]])] += 1

        self.emissions = np.zeros((len(self.all_words),len(self.all_tags)))

        for key,value in self.emissionsCount.items():
          
            denom = self.tagCounts[self.idx2tag[key[1]]]
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


        self.T = {key: set() for key in self.all_tags}
        self.T2 = {perm: set() for perm in permutations(self.all_tags, 2)}
        
        print(len(self.T2.keys()))

        for sentence in data[1]:
            for i in range(len(sentence)-1):
                self.T[sentence[i]].add(sentence[i+1])

        for sentence in data[1]: 
            for i in range(len(sentence)-2):
                if (sentence[i], sentence[i+1]) in self.T2: 
                    self.T2[(sentence[i], sentence[i+1])].add(sentence[i+2])

        ## TODO
        # count of each tag 
        for tag in self.all_tags: 
            self.tagCounts[tag] = 0
        for sentence in data[1]: 
            for tag in sentence: 
                self.tagCounts[tag] += 1

        
        self.N = sum(self.tagCounts.values())
        
        self.V = len(set(self.word2idx.keys()))
        
        self.get_unigrams()

        self.get_bigrams()

        self.get_trigrams()

        self.get_emissions()

        # Build the suffix mapping
        self.suffixes = defaultdict(Counter)
        for sentence, tag_seq in zip(data[0], data[1]):
            for word, tag in zip(sentence, tag_seq):
                # Here, we take the last 3 characters as the suffix; this number can be tuned
                self.suffixes[word[-3:]].update([tag])
                
        # Choose the most common tag for each suffix
        self.suffix_to_tag = {suffix: tags.most_common(1)[0][0] for suffix, tags in self.suffixes.items()}


        # Making the assumption that we are starting with bi-grams, n = 2. Can generalise later. 
        # Implement Smoothing -

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
        # ## TODO 

        if self.kgram == 2: 
            prev = 'O' # as this is start word
            tagSeq = ['O']
            for word in sequence[1:]: 
                # probability of word given the tag
                maxi = 0
                maxTag = ''

                if word in self.word2idx:
                    for tag2 in self.all_tags: 
                        q = self.bigrams[self.tag2idx[prev], self.tag2idx[tag2]]
                        e = self.emissions[self.word2idx[word], self.tag2idx[tag2]]
                        if q*e > maxi: 
                            maxTag = tag2
                            maxi = q*e
                    prev = maxTag
                    tagSeq.append(maxTag)
                else: 
                    if self.unknowns: 
                        # handling the unknown word as a noun
                        prev = 'NN'
                        tagSeq.append('NN')

                    else: # suffix tree mapping
                        prev = self.suffix_to_tag.get(word[-3:], "NN")  # default to noun if suffix not in mapping
                        tagSeq.append(prev)
            # print(tagSeq)
            return tagSeq
        elif self.kgram == 3: 
            prev1 = 'O' # as this is start word (the one just before current)
            prev2 = 'O' # the one 2 tags back
            tagSeq = ['O']
            for word in sequence[1:]: 
                # probability of word given the tag
                maxi = 0
                maxTag = ''
                # self.trigramsCount[(self.tag2idx[tag1], self.tag2idx[tag2],self.tag2idx[tag3])] = 0
            
                if word in self.word2idx:
                    for tag2 in self.all_tags: 
                        q = self.trigrams[self.tag2idx[prev2], self.tag2idx[prev1], self.tag2idx[tag2]]
                        e = self.emissions[self.word2idx[word], self.tag2idx[tag2]]
                        if q*e > maxi: 
                            maxTag = tag2
                            maxi = q*e
                    prev2 = prev1
                    prev1 = maxTag
                    tagSeq.append(maxTag)
                else: 
                    if self.unknowns: 
                        # handling the unknown word as a noun
                        prev = 'NN'
                        tagSeq.append('NN')

                    else: # suffix tree mapping
                        prev = self.suffix_to_tag.get(word[-3:], "NN")  # default to noun if suffix not in mapping
                        tagSeq.append(prev)
            # print(tagSeq)
            return tagSeq

    def beam(self, sequence, k):
        """ Tags a sequence with PoS tags

        Implements beam search"""
        
        ## TODO 
        if self.kgram == 2: 
            top_k_seq = [['O'] for i in range(k)]
            top_k_prob =  [0 for j in range(k)]
            for word in sequence[1:-1]: 
                # probSet = set()
                if word in self.word2idx:
                    cur_k_seq = {}
                    cur_k_seq_set = set()

                    # go through the all the sequences we have
                    for i in range(len(top_k_seq)): 
                        parent = top_k_seq[i]
                        parent_prob = top_k_prob[i]

                        # for each parent, check each possible tags
                        for tag in self.all_tags: 

                            q = self.bigrams[self.tag2idx[top_k_seq[i][-1]], self.tag2idx[tag]]
                            e = self.emissions[self.word2idx[word], self.tag2idx[tag]]
                            
                            if q*e == 0: 
                                # print(e)
                                continue

                            # for each tag for each sequence we have, calculate the prprobability
                            prob = log(q) + log(e) + parent_prob
                            
                            # keep adding sequences into the set so long as its length is less than k
                            # if length is k, then every time you add a sequence, remove the lowest prob sequence 
                            new_seq = copy.deepcopy(parent)
                            new_seq.append(tag)
                            # print(new_seq)
                            
                            # print(cur_k_seq_set)
                            
                            if tuple(new_seq) not in cur_k_seq_set:
                                if len(cur_k_seq_set) < k:
                                    cur_k_seq_set.add(tuple(new_seq))
                                    cur_k_seq[prob] = (new_seq, i)

                                # if the length is k or greater, repalce least probable element with new one(if higher)
                                else:                              
                                    minProb = min(cur_k_seq.keys())
                                    if prob > minProb: 
                                        cur_k_seq[prob] = (new_seq, i)
                                        remove_from_set = cur_k_seq[minProb]
                                        cur_k_seq_set.remove(tuple(remove_from_set[0]))
                                        cur_k_seq_set.add(tuple(new_seq))
                                        del cur_k_seq[minProb]
                
                    # we now have the top k or less probabilties based on uniqueness
                    # now, we must replace the original with these
                    top_k_prob = []
                    top_k_seq = []
                    
                    sorted_keys = sorted(cur_k_seq.keys(), reverse=True)
                    
                    for p in sorted_keys: 
                        top_k_seq.append(cur_k_seq[p][0])
                        top_k_prob.append(p)

                    # print(top_k_seq)
                                
                else: 
                    if self.unknowns: 
                        for seq in top_k_seq:
                            seq.append('NN')
                    else: 
                        for i in range(len(top_k_seq)):
                            cur_tag = self.suffix_to_tag.get(word[-3:], "NN")  # default to noun if suffix not in mapping
                            top_k_seq[i].append(cur_tag)
                            q = self.bigrams[self.tag2idx[top_k_seq[i][-1]], self.tag2idx[cur_tag]]
                            e = self.tagCounts[cur_tag]/self.N
                            top_k_prob[i] += log(q) + log(e)


            sol = top_k_seq[0]
            sol.append('.')
            return sol    
        elif self.kgram == 3:  
            top_k_seq = [['O','O'] for i in range(k)]
            top_k_prob =  [0 for j in range(k)]
            for word in sequence[1:-1]: 
                # probSet = set()
                if word in self.word2idx:
                    cur_k_seq = {}
                    cur_k_seq_set = set()

                    # go through the all the sequences we have
                    for i in range(len(top_k_seq)): 
                        parent = top_k_seq[i]
                        parent_prob = top_k_prob[i]

                        # for each parent, check each possible tags
                        for tag in self.all_tags: 

                            q = self.trigrams[self.tag2idx[top_k_seq[i][-2]], self.tag2idx[top_k_seq[i][-1]], self.tag2idx[tag]]

                            e = self.emissions[self.word2idx[word], self.tag2idx[tag]]
                            
                            if q*e == 0: 
                                # print(e)
                                continue

                            # for each tag for each sequence we have, calculate the prprobability
                            prob = log(q) + log(e) + parent_prob
                            
                            # keep adding sequences into the set so long as its length is less than k
                            # if length is k, then every time you add a sequence, remove the lowest prob sequence 
                            new_seq = copy.deepcopy(parent)
                            new_seq.append(tag)
                            # print(new_seq)
                            
                            # print(cur_k_seq_set)
                            
                            if tuple(new_seq) not in cur_k_seq_set:
                                if len(cur_k_seq_set) < k:
                                    cur_k_seq_set.add(tuple(new_seq))
                                    cur_k_seq[prob] = (new_seq, i)

                                # if the length is k or greater, repalce least probable element with new one(if higher)
                                else:                              
                                    minProb = min(cur_k_seq.keys())
                                    if prob > minProb: 
                                        cur_k_seq[prob] = (new_seq, i)
                                        remove_from_set = cur_k_seq[minProb]
                                        cur_k_seq_set.remove(tuple(remove_from_set[0]))
                                        cur_k_seq_set.add(tuple(new_seq))
                                        del cur_k_seq[minProb]
                
                    # we now have the top k or less probabilties based on uniqueness
                    # now, we must replace the original with these
                    top_k_prob = []
                    top_k_seq = []
                    
                    sorted_keys = sorted(cur_k_seq.keys(), reverse=True)
                    
                    for p in sorted_keys: 
                        top_k_seq.append(cur_k_seq[p][0])
                        top_k_prob.append(p)

                    # print(top_k_seq)
                                
                else: 
                    if self.unknowns: 
                        for seq in top_k_seq:
                            seq.append('NN')
                    else: 
                        for i in range(len(top_k_seq)):
                            cur_tag = self.suffix_to_tag.get(word[-3:], "NN")  # default to noun if suffix not in mapping
                            top_k_seq[i].append(cur_tag)
                            q = self.trigrams[self.tag2idx[top_k_seq[i][-2]], self.tag2idx[top_k_seq[i][-1]], self.tag2idx[cur_tag]]
                            e = self.tagCounts[cur_tag]/self.N
                            top_k_prob[i] += log(q) + log(e)

            sol = top_k_seq[0]
            sol.append('.')
            sol.pop(0)
            return sol       

    def viterbi (self, sequence):
        """ Tags a sequence with PoS tags

        Implements viterbi decoding"""

        # TODO
        # print(sequence)
        # this is for bigrams, another self.alltags for trigrams
        pi = np.ones((len(sequence), len(self.all_tags)))

        pi *= -math.inf
        for i in range(len(self.all_tags)): # assign the probs to 1 at the start tag for EACH tag
            pi[0,i] = 0

        bp = np.zeros((len(sequence), len(self.all_tags)))

        for i in range(1,len(sequence)): 

            word = sequence[i]
            # print(word)

            if word in self.word2idx: # if word is known
                # print(word)
                # print(pi[:5])
                
                for j in range(len(self.all_tags)): # go through each tag
                    tag_next = self.all_tags[j]

                    for k in range(len(self.all_tags)): # go through each tag
                        tag_prev = self.all_tags[k] 
                        
                        q = log(self.bigrams[self.tag2idx[tag_prev], self.tag2idx[tag_next]])
                        e = self.emissions[self.word2idx[word], self.tag2idx[tag_next]]
                        
                        if q*e != 0: 
                            e = log(e)
                        else: 
                            continue
                        
                        prod = q+e

                        prod += pi[i-1, k]
                        if prod > pi[i, j]: 
                            pi[i, j] = prod

                            # assign BP here
                            bp[i, j] = k

        
            else: # if word is unknown
                # print("Unkown: ", word)
                if self.unknowns: 
                    for j in range(len(self.all_tags)): # go through each tag
                        pi[i, j] = pi[i-1, j]
                        bp[i,j] = bp[i-1,j]

                else: # suffix tree mapping
                    for j in range(len(self.all_tags)): # go through each tag
                        tag_idx = self.tag2idx[self.suffix_to_tag.get(word[-3:], "NN")]  # default to noun if suffix not in mapping
                        max_prev_tag = self.all_tags[np.argmax(pi[i-1])]
                        q = self.bigrams[self.tag2idx[max_prev_tag], tag_idx]
                        e = self.tagCounts[self.idx2tag[tag_idx]]/self.N
                        pi[i, j] = pi[i-1, j] + log(q) + log(e)
                        bp[i,j] = self.tag2idx[max_prev_tag]    

        # Reconstruct the max sequence: 

        seq = ['' for i in range(len(sequence))]

        # Start with the last word's most probable tag
        seq[0] = 'O'

        seq[-1] = self.idx2tag[np.argmax(pi[-1])]

        # Backtrack through the rest of the sequence
        for i in range(len(sequence)-2, 0, -1):
            max_idx = np.argmax(pi[i+1])
            seq[i] = self.idx2tag[int(bp[i+1, max_idx])]
        
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