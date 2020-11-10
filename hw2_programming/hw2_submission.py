#!/usr/bin/python

import random
import collections
import math
import sys
from util import *

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    c = collections.Counter(x.split(' '))
    return c
    # END_YOUR_CODE

############################################################
# Problem 3b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)
    def predict(x):
        phi = featureExtractor(x)
        dp = dotProduct(weights,phi)
        if dp < 0:
            return -1
        else:
            return 1    

    for i in range(numIters):
        for item in trainExamples:
            x,y = item
            phi = featureExtractor(x)
            dp = dotProduct(weights,phi)*y
            if dp < 1:
                increment(weights,-eta*-y,phi)
        print("Iteration:%s, Training error:%s",i,evaluatePredictor(trainExamples,predict),evaluatePredictor(testExamples,predict))

    # END_YOUR_CODE
    return weights

############################################################
# Problem 3c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        phi = dict() #key = subset of key in weight
        keys = random.randint(1,len(weights))
        for item in random.sample(list(weights), keys):
            phi[item] = random.randint(1,50)
        score = dotProduct(weights,phi)
        if score > 0: 
            y = 1
        else:
            y = -1
        
        # END_YOUR_CODE
        return (phi, y)
    return [generateExample() for _ in range(numExamples)]

############################################################
# Problem 3e: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
        # BEGIN_YOUR_CODE (our solution is 8 lines of code, but don't worry if you deviate from this)
        result = dict()
        x = x.replace(" ","")
        for j in range(0,len(x)-n+1):
           word = x[j:j+n]
           if word in result:
               result[word] += 1
           else:
               result[word] = 1
        return result
        # END_YOUR_CODE
    return extract

############################################################
# Problem 4: k-means
############################################################
def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 32 lines of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")

    newSqLoss = 0.0
    sqLoss = 9999999999999999 / 2
    z = [0 for _ in range(len(examples))] #z
    nearestCenter = [0.0 for _ in range(K)]
    centers = [sample.copy() for sample in random.sample(examples,K)] #u 
    
    for _ in range(maxIters):
        newSqLoss = 0.0
        
        #find the center by calculating the distance
        for i in range(0, len(centers)):
            nearestCenter[i] = float(dotProduct(centers[i], centers[i]))

        #classify each points to the nearest center
        for i in range(0, len(examples)):
            distance = dict()
            for j in range(0, len(centers)):
                temp = nearestCenter[j]
                for keys, values in examples[i].items():
                     #calculate the loss mean
                     temp += values * values - 2 * centers[j].get(keys, 0) * values
                distance[j] = temp
            #find the min distance by looping through the dictionary
            minDistance = min(distance.keys(), key = (lambda k: distance[k]))
            #set z given centroid
            z[i] = minDistance
            newSqLoss += distance[minDistance]

        #check if it converges
        #break if true
        if ((sqLoss - newSqLoss) <= 0):
            break
        if (sqLoss - newSqLoss) <= 0.01 * newSqLoss:
            break 
        sqLoss = newSqLoss

        # calculate new centers
        cluster = [0 for _ in range(K)]
        for i in range(0, len(centers)):
            centers[i].clear()
 
        #set centroids given z
        for i in range(0, len(examples)):
            increment(centers[z[i]], 1.0, examples[i])
            cluster[z[i]] += 1

        for i in range(0, len(centers)):
            avg = centers[i]
            #set average centroids to average of points assigned to cluster k
            for item in avg:
                avg[item] = avg[item] * 1/float(cluster[i])

    return (centers, z, newSqLoss)
    # END_YOUR_CODE
