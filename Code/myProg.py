from __future__ import division
from optparse import OptionParser
from math import log

from gensim import corpora, models, similarities
from collections import defaultdict
import logging
import gensim

import sys
import math
import random
import subprocess
import sys,re
######################LDA##################
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
###############Term-Document Matrix######################################
def termFrequency(term, document):
    normalizeDocument = document.lower().split()
    return normalizeDocument.count(term.lower()) / float(len(normalizeDocument))

def inverseDocumentFrequency(term, allDocuments):
    numDocumentsWithThisTerm = 0
    for doc in range(len(allDocuments)):
        if term.lower() in allDocuments[doc].lower().split():
            numDocumentsWithThisTerm = numDocumentsWithThisTerm + 1
 
    if numDocumentsWithThisTerm > 0:
        return 1.0 + log(float(len(allDocuments)) / numDocumentsWithThisTerm)
    else:
        return 1.0
def getVocab(allDocuments):
    wcount=0
    wdict={}
    i = 0
    for doc in allDocuments:
    	    doc = doc.strip("\n")
            wl = doc.split(" ")
            for w in wl:
                if not (w in wdict.keys()):
                    wcount += 1
                    wdict[w] = i
                    i += 1
       
    return (wcount, wdict)
    
def saveModelParams():
    f = open("intermediate.txt","r").readlines()
    f2 = open("modelParams.txt","w")
    fdict = open("worddict.txt","w")
    (vc, wdict) = getVocab(f)
    #map of sentences/docs/commentaries
    D={}
    i=0
    for d in f:
    	    d = d.strip("\n")
            D[d] = i; 
            i+=1
    fdict.write(str(vc)+"\n")
    for key,value in wdict.items():
    	key=key.strip("\n")
        fdict.write(str(key)+" "+str(value)+"\n")
    fdict.close()
    termdocmatrix=[[0.000000 for j in range(vc)] for i in range(len(f))]
    for line in f:
    	line = line.strip("\n")
        wl = line.split()
        for w in wl:
            tf = termFrequency(w,line)
            idf = inverseDocumentFrequency(w,f)
            tfidf = tf*idf
            termdocmatrix[D[line]][wdict[w]] = tfidf
    
    for i in range(len(termdocmatrix)):
        for j in range(len(termdocmatrix[i])):
     	      f2.write("%.6f "%(100*termdocmatrix[i][j]))
        f2.write("\n")
    f2.close()
######################End Term-Document Matrix###############################################
#########################K-Means Clustering##################################################
class Point:
    '''
    An point in n dimensional space
    '''
    def __init__(self, coords, id1=None):
        '''
        coords - A list of values, one per dimension
        '''
        self.id = id1
        self.coords = coords
        self.n = len(coords)
        
    def __repr__(self):
        return str(self.coords)

class Cluster:
    '''
    A set of points and their centroid
    '''
    
    def __init__(self, points):
        '''
        points - A list of point objects
        '''
        
        if len(points) == 0: raise Exception("ILLEGAL: empty cluster")
        # The points that belong to this cluster
        self.points = points
        
        # The dimensionality of the points in this cluster
        self.n = points[0].n
        
        # Assert that all points are of the same dimensionality
        for p in points:
            if p.n != self.n: raise Exception("ILLEGAL: wrong dimensions")
            
        # Set up the initial centroid (this is usually based off one point)
        self.centroid = self.calculateCentroid()
        
    def __repr__(self):
        '''
        String representation of this object
        '''
        return str(self.points)
    
    def update(self, points):
        '''
        Returns the distance between the previous centroid and the new after
        recalculating and storing the new centroid.
        '''
        old_centroid = self.centroid
        self.points = points
        self.centroid = self.calculateCentroid()
        shift = getDistance(old_centroid, self.centroid) 
        return shift
    
    def calculateCentroid(self):
        '''
        Finds a virtual center point for a group of n-dimensional points
        '''
        numPoints = len(self.points)
        # Get a list of all coordinates in this cluster
        coords = [p.coords for p in self.points]
        # Reformat that so all x's are together, all y'z etc.
        unzipped = zip(*coords)
        # Calculate the mean for each dimension
        centroid_coords = [math.fsum(dList)/numPoints for dList in unzipped]
        
        return Point(centroid_coords)

def kmeans(points, k, cutoff):
    
    # Pick out k random points to use as our initial centroids
    initial = random.sample(points, k)
    
    # Create k clusters using those centroids
    clusters = [Cluster([p]) for p in initial]
    
    # Loop through the dataset until the clusters stabilize
    loopCounter = 0
    while True:
        # Create a list of lists to hold the points in each cluster
        lists = [ [] for c in clusters]
        clusterCount = len(clusters)
        
        # Start counting loops
        loopCounter += 1
        # For every point in the dataset ...
        for p in points:
            # Get the distance between that point and the centroid of the first
            # cluster.
            smallest_distance = getDistance(p, clusters[0].centroid)
        
            # Set the cluster this point belongs to
            clusterIndex = 0
        
            # For the remainder of the clusters ...
            for i in range(clusterCount - 1):
                # calculate the distance of that point to each other cluster's
                # centroid.
                distance = getDistance(p, clusters[i+1].centroid)
                # If it's closer to that cluster's centroid update what we
                # think the smallest distance is, and set the point to belong
                # to that cluster
                if distance < smallest_distance:
                    smallest_distance = distance
                    clusterIndex = i+1
            lists[clusterIndex].append(p)
        
        # Set our biggest_shift to zero for this iteration
        biggest_shift = 0.0
        
        # As many times as there are clusters ...
        for i in range(clusterCount):
            # Calculate how far the centroid moved in this iteration
            shift = clusters[i].update(lists[i])
            # Keep track of the largest move from all cluster centroid updates
            biggest_shift = max(biggest_shift, shift)
        
        # If the centroids have stopped moving much, say we're done!
        if biggest_shift < cutoff:
            print "Converged after %s iterations" % loopCounter
            break
    return clusters

def getDistance(a, b):
    '''
    Euclidean distance between two n-dimensional points.
    Note: This can be very slow and does not scale well
    '''
    if a.n != b.n:
        raise Exception("ILLEGAL: non comparable points")
    
    ret = reduce(lambda x,y: x + pow((a.coords[y]-b.coords[y]), 2),range(a.n),0.0)
    return math.sqrt(ret)

def makeRandomPoint(n, lower, upper):
    '''
    Returns a Point object with n dimensions and values between lower and
    upper in each of those dimensions
    '''
    p = Point([random.uniform(lower, upper) for i in range(n)],i in range(n))
    return p


def loadModelParams():
    f = open("modelParams.txt","r").readlines()
    #num_points =  #Text commentaries in our dataset
    
    num_points = len(f)
    
    # dimension of each point = vocabulary size
    line = f[0].strip("\n")
    wl = line.split()
    dimensions = len(wl)
    
    # Bounds for the values of those points in each dimension
    lower = 0
    upper =100
    # The K in k-means.We have obtained optimal number of clusters as 10 by elbow method,as discussed in report.
    num_clusters = 10
    
    # optimization has 'converged' after the max distance between clusters do not differ more than threshold, stop updating clusters
    opt_cutoff = 0.5
    
    # Generate some points as random seed for our clustering
    points = [makeRandomPoint(dimensions, lower, upper) for i in xrange(num_points)]
    i1=0
    for line1 in f:
        line1 = line1.strip("\n")
        wl1= line1.split()
        wl2 = [0.000000 for i2 in range(len(wl1))]
        for i3 in range(len(wl1)):
            nt=float(wl1[i3])
            wl2[i3] = nt
        if not wl2:
        	continue
        points[i1]=Point(wl2,i1)
        i1+=1
        
    # perform clustering
    clusters = kmeans(points, num_clusters, opt_cutoff)
    fclout = open("clustersWithPointId.txt","w")
    sdict = {}
    fs= open("intermediate.txt","r").readlines()
    for id1 in range(len(fs)):
        sdict[id1] = fs[id1]
    
    # Print clusters
    for i5,c in enumerate(clusters):
        for p in c.points:
            fclout.write(str(i5)+" ")
            fclout.write("%d "%int(p.id))
            fclout.write(sdict[int(p.id)])
        fclout.write("\n")
    fclout.close()
    # Estimate SSE error
    sse =0.00
    for cid,c in enumerate(clusters):
    	cd =0.00
        for p in c.points:
    		d =getDistance(p,clusters[cid].centroid)
    		cd =cd + d
    	sse+=cd
    print("sse "+str(sse))
   
####################End K-Means Clustering#################################################
###################LDA#################################

def LDA():
	f=open("clustersWithPointId.txt").readlines()
	#forig = open(fCorpus,"r").readlines()
	fresult=open("result.txt","w")
	documentss=[ [] for i in range(len(f))]
	docid=0
	for line in f:
		l = ""
		line = line.split()
		if not line:
			continue
		docid = int(line[0])
		for k in range(2,len(line)):
			l+=line[k]
			l+=" "
		l+="\n"
		documentss[docid].append(l)
		
	# remove common words and tokenize
	stoplist = set('for a of the and to in'.split())
	c=0
	docidt=0
	for documents in documentss:
		if not documents:
			continue
		texts = [[word for word in document.lower().split() if word not in stoplist]
		for document in documents]

		dictionary = corpora.Dictionary(texts)



		corpus = [dictionary.doc2bow(text) for text in texts]

		lda =models.LdaModel(corpus, id2word=dictionary, num_topics=5, distributed=False)
		fresult.write("cluster"+str(c)+"\n")
		for doc in documents:
			fresult.write(doc)
		#for i in range(docidt,docidt+len(documents)):	
		#	fresult.write(str(i)+" "+ forig[i])
		#docidt+=len(documents)
		#print 'Beta K, K=5'
		for x in range(0,5):
			fresult.write( '\ntopic %d :' %(x))
			fresult.write(lda.print_topic(x))
		fresult.write("\n-------------------------------------------------------------------------------\n")
		c+=1
#############end LDA###############
def main():
	saveModelParams()
	loadModelParams()
	LDA()
if __name__ == "__main__":
	main()
