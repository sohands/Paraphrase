import pickle,os,random
from collections import Counter
import numpy

os.mkdir('walks_extension_sigmoid')

def randomWalkWithSource(M,source,iterations,restart):
	current = source
	length = len(M[0])
	seq = [source]
	for i in xrange(iterations):
		probs = M[current]
		restart_prob = random.random()
		if restart_prob < restart :
			current = source
			seq.append(source)
			continue
		nextNode = numpy.random.choice(length,1,p=probs)[0]
		#print nextNode
		#r = random.random()
		#for nextNode in xrange(length):
		#	if r < sum(probs[:nextNode]):
		#		break
		seq.append(nextNode)
		current = nextNode
	return seq

def getRandomWalk(M):
	P = []
	damping = 0.5 # value for damping
	eps = 0.0000000000001	# the error difference, which should ideally be zero but can never be attained.
	restart = 0.5		# restart probability
	iterations = 5000
	M *= damping
	M += ((1.0-damping)/(1.0*len(M[0])))
	length = len(M[0])
	for node in xrange(length):
		v = []
		seq = randomWalkWithSource(M,node,iterations,restart)
		freq = dict(Counter(seq))
		total_visits = len(seq)
		for n in xrange(length):
			if n not in freq.keys():
				freq[n]=0
			visit_prob = float(freq[n])/float(total_visits)
			v.append(visit_prob)
		P.append(v)
		print 'Walked for node ',node
	return P

def getMatrix(dimension):
	return pickle.load(open('matrices_extension_sigmoid/'+str(dimension)+'.pickle','r'))

for d in xrange(678):
	M=getMatrix(d)
	print 'Loaded for ',d
	P=getRandomWalk(M)
	print 'Walked completed for ',d
	f=open('walks_extension_sigmoid/'+str(d)+'.pickle','w')
	pickle.dump(P,f)
	f.close()
	print 'Dumped for ',d
