import pandas as pd
from sklearn import preprocessing
import math
import numpy as np


# File should be organized in lines with every feature first, and then corresponding class
# it changes the classes to an integer interval from 0 to N-1
def loadDataSetFromFile(file,columClasses="last"):

	dataset = pd.read_csv(file)
	dimensions = dataset.shape
	numColumns = dimensions[1]
	if columClasses == "last":
		X = dataset.ix[:,:-1]
		y = dataset.ix[:,-1]
	else:
		X = dataset.ix[:,1:]
		y = dataset.ix[:,0]
	le = preprocessing.LabelEncoder()
	le.fit(y)
	y = le.transform(y)
	return X,y

def selectNRandomColumns(instances,numberOfFeatures):
	dimensions = instances.shape
	totNumFeatures = dimensions[1]
	numbers = []
	for i in range(totNumFeatures):
		numbers.append(i)
	numbersRand = np.random.permutation(numbers)
	indexNumbers = numbersRand[0:numberOfFeatures]
	instancesNumPy = np.array(instances)
	return instancesNumPy[:,indexNumbers]


def divideDataSetInPartitions(instances):
	dimensions = instances.shape
	numFeatures = dimensions[1]
	ret = {}
	instancesNumPy = np.array(instances)
	n = min(5, numFeatures/2) # as explained in the article, the number of local agents will be 5
	numbers = []
	for i in range(numFeatures):
		numbers.append(i)
	numbersRand = np.random.permutation(numbers)
	if (n <= numFeatures):
		numberOfFeaturesInEachModel = int( math.ceil (numFeatures / n))
		for i in range(n):
			startIndex = i*numberOfFeaturesInEachModel
			endIndex = min((i+1) * numberOfFeaturesInEachModel,numFeatures)
			indexNumbers = numbersRand[startIndex:endIndex]
			extractedData = instancesNumPy[:,indexNumbers]
			ret.update( {i : extractedData } )
		return ret
	else:
		print "ERROR, cannot put in a number of partitions greater than the number of features"
		return None