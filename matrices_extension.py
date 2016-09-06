import numpy as np
from scipy.stats import spearmanr
import math

def sigmoid(x):
    return 1/(1+math.exp(-x))

def getCorrelation(pred,gold):
    return spearmanr(pred,gold)[0]

def lookup(We,words,w):
    w = w.lower()
    if len(w) > 1 and w[0] == '#':
        w = w.replace("#","")
    if w in words:
        return We[words[w],:],False
    else:
        return We[words['UUUNKKK'],:],True

def read_data(file):
    file = open(file,'r')
    lines = file.readlines()
    lines.pop(0)
    examples = []
    for i in lines:
        i=i.strip()
        i=i.lower()
        if(len(i) > 0):
            i=i.split()
            ex = (i[0],i[1],float(i[2]))
            examples.append(ex)
    return examples

def getAllwords(ex):
    all_words = []
    for e in ex:
        all_words.append(e[0])
        all_words.append(e[1])
    return all_words

def getWordmap(textfile):
    words={}
    We = []
    f = open(textfile,'r')
    lines = f.readlines()
    for (n,i) in enumerate(lines):
        i=i.split()
        j = 1
        v = []
        while j < len(i):
            v.append(float(i[j]))
            j += 1
        words[i[0]]=n
        We.append(v)
    return (words, np.array(We))

(words, We) = getWordmap('../data/glove_small.txt')
ws353ex = read_data('../data/wordsim353.txt')

all_words = getAllwords(ws353ex)        
all_words = list(set(all_words))

length = len(all_words)

cutoff = 7.5

similar_ex = []

for ex in ws353ex:
    if ex[2] > cutoff:
        similar_ex.append(ex)

dimension_list = []
correlations = []

word_index = {}
word_index_reverse = {}

counter = 0
for word in all_words:
    word_index[counter] = word
    word_index_reverse[word] = counter
    counter+=1

for i in xrange(300):
    list1 = []
    list2 = []
    for ex in similar_ex:
        v1,u1 = lookup(We,words,ex[0])
        v2,u2 = lookup(We,words,ex[1])
        list1.append(v1[i])
        list2.append(v2[i])
    correlations.append(getCorrelation(list1,list2))
    # print 'Done for',i

# print correlations

corr_cutoff = 0.55


for i in xrange(len(correlations)):
    if correlations[i] > corr_cutoff:
        dimension_list.append(i)

tuples = []

for d in dimension_list:
    for d2 in dimension_list:
        if d!=d2 and (d2,d) not in tuples:
            tuples.append((d,d2))

print len(tuples)+300

import os,pickle
os.mkdir('matrices_extension_sigmoid')

for dimension in xrange(300):
    M=np.zeros(shape=(length,length))
    for i in xrange(length):
        w1 = all_words[i]
        l1,u1= lookup(We,words,w1)
        #print l1
        for j in xrange(length):
            w2 = all_words[j]
            l2,u2= lookup(We,words,w2)
            #print l2
            M[i][j] = sigmoid(l1[dimension] - l2[dimension])
        # normalize rows
        sum=0
        for j in xrange(length):
            sum=sum+M[i][j]

        for j in xrange(length):
            M[i][j]=M[i][j]/sum 

    print 'Made for ',dimension
    f=open('matrices_extension_sigmoid/'+str(dimension)+'.pickle','w')
    pickle.dump(M,f)
    f.close()
    print 'Dumped for ',dimension

for dimension in xrange(len(tuples)):
    M=np.zeros(shape=(length,length))
    for i in xrange(length):
        w1 = all_words[i]
        l1,u1= lookup(We,words,w1)
        #print l1
        for j in xrange(length):
            w2 = all_words[j]
            l2,u2= lookup(We,words,w2)
            #print l2
            d1 = tuples[dimension][0]
            d2 = tuples[dimension][1]
            a = l1[d1] - l2[d1]
            b = l1[d2] - l2[d2]
            M[i][j] = sigmoid(a-b)
        # normalize rows
        sum=0
        for j in xrange(length):
            sum=sum+M[i][j]

        for j in xrange(length):
            M[i][j]=M[i][j]/sum 

    print 'Made for ',dimension+300
    f=open('matrices_extension_sigmoid/'+str(dimension+300)+'.pickle','w')
    pickle.dump(M,f)
    f.close()
    print 'Dumped for ',dimension+300
