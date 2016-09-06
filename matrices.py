
import numpy as np
import math

def sigmoid(x):
    return 1/(1+math.exp(-x))

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

import os,pickle
os.mkdir('matrices_new_sigmoid')

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
    f=open('matrices_new_sigmoid/'+str(dimension)+'.pickle','w')
    pickle.dump(M,f)
    f.close()
    print 'Dumped for ',dimension


