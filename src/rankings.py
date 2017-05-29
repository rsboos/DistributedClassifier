import numpy as np

def makeRankingForClass(data):
	index=0
	tmp = []
	ret = []
	for elem in data:
		tmp.append((elem,index))
		index+=1
	tmp.sort(key=lambda tup: tup[0], reverse=True)
	for elem in tmp:
		ret.append(elem[1])
	return ret

def makeRankingsForModel(probabilitiesVector):
	nClasses = probabilitiesVector.shape[1] #number of Columns is equal to the number of classes
	ret = {}
	for classIndex in range(nClasses):
		classData = probabilitiesVector[:,classIndex]
		rankClass = makeRankingForClass(classData)
		ret.update( {classIndex: rankClass} )
	return ret

def makeRankings(probabilitiesDict):
	ret = {}
	for index, probs in probabilitiesDict.items():
		rankingsForModel = makeRankingsForModel(probs)
		ret.update( {index: rankingsForModel } )
	return ret