import pickle,os
import numpy as np
from sklearn import datasets,linear_model
from scipy.stats import spearmanr
from random import shuffle

# os.mkdir('average')

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

# print 'Got all words'

word_sim = {}
word_index = {}
word_index_reverse = {}
counter = 0

for word in all_words:
	word_sim[word] = np.zeros(shape=(1,len(all_words)))
	word_index[counter] = word
	word_index_reverse[word] = counter
	counter+=1

for i in xrange(678):
	M=pickle.load(open('walks_extension_sigmoid/'+str(i)+'.pickle','r'))
	for j in xrange(len(M)):
		word = word_index[j]
		word_sim[word] += np.asmatrix(M[j])
	# print 'Done for ',i

# f=open('average/similarity.txt','w')
pred = []
gold = []

for ex in ws353ex:
	# print 'Finding for ',ex[0],'-',ex[1],'\t',word_index_reverse[ex[1]],word_sim[ex[0]].shape
	sim1=word_sim[ex[0]][0][word_index_reverse[ex[1]]]
	sim2=word_sim[ex[1]][0][word_index_reverse[ex[0]]]
	pred.append(sim1+sim2)
	gold.append(ex[2])
	# f.write(ex[0]+'\t'+ex[1]+'\t'+str(ex[2])+'\t'+str(sim1)+'\t'+str(sim2)+'\n')

corr = getCorrelation(pred,gold)
print 'Average Correlation :',corr
# f.write('Correlation : '+str(corr)+'\n')
# f.close()

# os.mkdir('vectors')
# f=open('vectors/word_vectors.pickle','w')
# pickle.dump({'vectors':word_sim,'indices':word_index,'reverse_indices':word_index_reverse},f)
# f.close()

# os.mkdir('learned')

# print 'Learning weights'

models = {'Ordinary Regression' : linear_model.LinearRegression()}
alphas = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
for a in alphas:
    models['Ridge Regression with alpha = '+str(a)]=linear_model.Ridge(alpha = a)

X_train = []
X_test = []
y_train = []
y_test = []
shuffle(ws353ex)
train = ws353ex[:int(0.9*len(ws353ex))]
test = ws353ex[int(0.9*len(ws353ex)):]

for ex in train :
    X_train.append([])
    y_train.append(ex[2])

for ex in test:
    X_test.append([])
    y_test.append(ex[2])

for a in xrange(678):
    M = pickle.load(open('walks_extension_sigmoid/'+str(a)+'.pickle','r'))
    counter = 0
    for ex in train:
        i = word_index_reverse[ex[0]]
        j = word_index_reverse[ex[1]]
        v1 = M[i]
        v2 = M[j]
        X_train[counter].append(v1[j]+v2[i])
        counter += 1
    counter = 0
    for ex in test:
        i = word_index_reverse[ex[0]]
        j = word_index_reverse[ex[1]]
        v1 = M[i]
        v2 = M[j]
        X_test[counter].append(v1[j]+v2[i])
        counter += 1
# print 'Prepared dataset'
os.mkdir('models')
for m in models.keys():
    models[m].fit(X_train,y_train)
    pred_train = models[m].predict(X_train)
    pred_test = models[m].predict(X_test)
    print m
    print '\t\tTrain :',getCorrelation(pred_train,y_train)
    print '\t\tTest :',getCorrelation(pred_test,y_test)

pickle.dump(models,open('models/models_extension_sigmoid.pickle','w'))
