#-----------------------------------------------------------------------------------
#   Multinomial Naїve Bayes' Classifier v.0.0.1
#
#        Pr,Cs = compute(D,C,S)
#
#        D - set of documents, C - set of classes, S - input sample
#
#        Pr - the posteriors of each class in C, Cs - the class of the sample S
#
#        The worst-case complexity of the Multinomial Bayes' classifier:
#
#                   p = O(nmz + 2nm), where n - # number of terms in S, 
#                                           m - # of classes in C
#                                           z - # of documents in D
#
#                   An Example: n = 100, m = 15, z = 10^4 => p ~ O(1,5e+07)
#
#   GNU Public License (C) 2021 Arthur V. Ratz
#-----------------------------------------------------------------------------------

import os
import re
import csv
import math
import nltk
import numpy as np

def log_p(n):
    # Get the probability's n natural algorithm
    return abs(math.log(abs(n))) \
        if n != 0 and n != 1 else 1

def get_fea_class(D,k):
    # Get all documents in D belonging to the k-th class in C
    return np.array([ d for d in D if k == int(d[0]) ])

def get_count_class(D,k):
    # Get the count p(Ck) of the class Ck in documents D
    return len(get_fea_class(D, k))

def get_counts_term(D,w):
    # Get the count of the term w occurrences in each document from D
    count_wt = np.array([ len([ term \
        for term in d[1] if w == term ]) for d in D ])
    # Get the total count of documents from D, containing the term w
    return len(np.array([ f_wt \
        for f_wt in count_wt if f_wt > 0 ]))

def get_prob_class(D,k):
    # Get the probability p(Ck) of the k-th class Ck
    return get_count_class(D,k) / len(D)

def get_probs_term(D,w):
    # Get the probability of the term w occurrence 
    # in each document from the class Ck
    return get_counts_term(D,w) / len(D)

def parse(S):
    W = S.lower().split()

    # Parse the string S, performing 
    # the normalization and word-stamming using NLTK library
    W = np.array([ re.sub(r"""[,.;@#?!&$\']+\ *""", '', w) for w in W])
    W = np.array([ tag[0] for tag in nltk.pos_tag(W) \
        if re.match('NN', tag[1]) != None or re.match('JJ', tag[1]) != None ])

    return np.array([ w for w in W if len(w) > 2 ])
    
def build_model(D):
    # Build the class prediction model, 
    # based on the corpus of documents in D
    D = np.array([ np.array([ d[0], parse(d[1]) ], \
        dtype=object) for d in D ], dtype=object)
    return np.array([ d for d in D if len(d[1]) > 0 ])

def compute(D,C,S):
    W = parse(S);                 # A set of terms W in the sample S
    Pr = np.empty(0);             # A set of posteriors Pr(Ck | W)

    n = len(W); m = len(C)        # n - # of terms W in S
                                  # m - # of classes in C

    # For each k-th class Ck, compute the posterior Pr(Ck | W)
    for k in range(m):
        pr_ck_w = 0                  # pr_ck_w - the likelihood P(Ck | wi) 
                                     # of Ck is the class of the term wi

        d_ck = get_fea_class(D,k)    # d_ck - A set of documents from the class Ck
        p_ck = get_prob_class(D,k)   # p_ck - Probability of the k-th class Ck in documents D

        # For each term W[i], compute the likelihood P(Ck | wi)
        for i in range(n):
            # Obtain the count and probability of the 
            # term W[i] in the documents from class Ck
            prob_wd_n = get_probs_term(d_ck, W[i])
            count_wt_n = get_counts_term(d_ck, W[i])
            
            pr_ck_w += count_wt_n * \
                log_p(prob_wd_n) if count_wt_n > 0 else 0

        pr_ck_w += p_ck

        # Append the posterior Pr(Ck | W) of the class Ck to the array Pr
        Pr = np.append(Pr, pr_ck_w)

    # Obtain an index of the class Cs as the class in C, 
    # having the maximum posterior Pr(Ck | W)
    Cs = np.where(Pr == np.max(Pr))[0][0]
   
    return Pr,Cs   # Return the array of posteriors Pr
                   # and the index of sample S class Cs

def evaluate(T,D,C):
    print('Classification:')
    print('===============\n')

    # For each sample S in the set T, compute the class of S
    # Estimate the real classification's multinomial entropy and its expectation
    for s in T[:,1]:
        pr_s = '\0'; \
            Pr,Cs = compute(D,C,s)
        for ci,p in zip(range(len(C)),Pr):
            pr_s += prob_stats % (C[ci][1],p)

        print(sampl_stats % (s, C[Cs][1] \
            if np.sum(Pr) > 0 else 'None', pr_s))

def load_data(filename):
    cols_max = 2; \
       data = np.empty((0, cols_max))
    filename = '..\\dataset\\' + filename
    filename = os.path.dirname( \
        os.path.realpath(__file__)) + '\\' + filename
    with open(filename, newline='\n') as csvfile:
        for line in csv.reader(csvfile, delimiter='_', \
            quotechar='', quoting=csv.QUOTE_NONE):
                data = np.append(data, [line], axis=0)

    return np.array(data)

def output_data(T,D,C,fmt):

    print(model_stats % \
        (len(C),len(D),len(T)))

    print('Classes:')
    print('========\n')

    for c in C:
        k = int(c[0])
        dc = get_fea_class(D,k); \
            p_ck = get_count_class(D,k) / len(D)
        pd_stats = fmt % (len(dc), k + 1, p_ck)
        print('C%d: %s %s' % (k + 1, \
            '{0: <12}'.format(c[1]), pd_stats))

    print('\n')

    print('Documents:')
    print('==========\n')

    for d in D:
        print('C%d: \"%s...\"' % \
            (int(d[0]) + 1, d[1][:80]))

    print("\n")
 
prob_stats  = 'Pr(%s) = %f '
class_stats = '[ Documents: %3d, P(C%d) = %f ]'
sampl_stats = 'Text: [ \"%s\" ]\nClass: \"%s\" [%s]\n'
model_stats = '[ Classes: %d Documents: %d Samples: %d ]\n'
multi_stats = 'Multinomial Entropy: [ max: %f, real: %f ] classes/term\n'

app_banner  = '\nMultinomial Naїve Bayes\' Classifier | GNU License (C) 2021 | Arthur V. Ratz'

def main():
    print(app_banner)
    print('===========================================================================\n')

    T = load_data('eval.txt'); \
        D = load_data('trainset.txt'); \
            C = load_data('classes.txt')

    M = build_model(D)

    output_data(T,D,C,class_stats); \
        entropy = evaluate(T,M,C);

if __name__ == '__main__':
    main()