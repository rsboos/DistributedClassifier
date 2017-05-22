import numpy as np
from sklearn import preprocessing
import pandas as pd
import math
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.naive_bayes import GaussianNB
import sklearn.svm


# File should be organized in lines with every feature first, and then corresponding class
# it changes the classes to integer intervals from 0 to N-1
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

def divideDataSetInNPartitions(instances,n):
	dimensions = instances.shape
	numFeatures = dimensions[1]
	ret = {}
	instancesNumPy = np.array(instances)
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

form: high prob - index
def makeRankingsForModel(individualsProbs):
	numberOfClasses = individualsProbs.shape[1]
	for i in range(numberOfClasses):
		probsForClass = individualsProbs[:,[i]]
		for (elem)
			elem = (elem,index)
		sorted(arrayOFtuPLES, key=lambda x: x[0 or 1])


def makeRankings(outProbs,nModels):
	for i in range(nModels):
		makeRankingsForModel(outProbs[i])

def agregateRankings():
	for each classe
		n rankings

def calculateBordaCount():

def calculateCopelandFunction():

def executeAlgorithm(dataSetFile,numberOfModels):

	instancesFeatures, instancesClasses = loadDataSetFromFile(dataSetFile)
	modelsData = divideDataSetInNPartitions(instancesFeatures,numberOfModels)

	outputProbabilities = {}
	for i in range(numberOfModels):
		if (i == 0):
			vectorProbs = OneVsRestClassifier(sklearn.svm.SVC(probability=1)).fit(modelsData[i], instancesClasses).predict_proba(modelsData[i])
		elif (i == 1):
			vectorProbs = sklearn.svm.SVC(decision_function_shape='ovo',probability=1).fit(modelsData[i], instancesClasses).predict_proba(modelsData[i])
		outputProbabilities.update( {i : vectorProbs } )
	print outputProbabilities
	makeRankings(outputProbabilities,numberOfModels)

	agregateRankings()




	
def main():
    executeAlgorithm("../datasets/iris.data.txt",2)

if __name__ == "__main__":
    main()