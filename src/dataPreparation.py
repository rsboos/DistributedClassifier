import pandas as pd
from sklearn import preprocessing
import math
import numpy as np


# File should be organized in lines with every feature first, and then corresponding class
# it changes the classes to an integer interval from 0 to N-1
def loadDataSetFromFile(file):
	dataset = pd.read_csv(file)
	dimensions = dataset.shape
	numColumns = dimensions[1]
	X = dataset.ix[:,:-1]
	y = dataset.ix[:,-1]
	le = preprocessing.LabelEncoder()
	le.fit(y)
	y = le.transform(y)
	return X,y

def divideDataSetInPartitions(instances):
	dimensions = instances.shape
	numFeatures = dimensions[1]
	ret = {}
	instancesNumPy = np.array(instances)

	n = min(5, numFeatures/2) # as explained in the article, the number of local agents will be 5
	
	if (n <= numFeatures):
		numberOfFeaturesInEachModel = int( math.ceil (numFeatures / n))
		for i in range(n):
			startIndex = i*numberOfFeaturesInEachModel
			endIndex = min((i+1) * numberOfFeaturesInEachModel,numFeatures)
			extractedData = instancesNumPy[:,startIndex:endIndex]
			ret.update( {i : extractedData } )
		return ret
	else:
		print "ERROR, cannot put in a number of partitions greater than the number of features"
		return None