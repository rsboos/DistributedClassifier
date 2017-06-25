import dataPreparation
import classifier
import rankings
import socialChoiceEstimator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np


class FacilitatorAgent:
	def __init__(self, dataSetFile, classesPlace, kFolds):
		self.dataSetFile=dataSetFile
		self.instancesFeatures, self.instancesClasses = dataPreparation.loadDataSetFromFile(self.dataSetFile,classesPlace)
		self.numberOfModels=0
		self.algorithmsIndex = { "SVM": 0, "DecisionTree": 1, "KNN": 2, "NN": 3, "NB": 4, "ECOC": 5}
		self.kFolds = kFolds

	def execute(self,executionType="distributed",function=None,numberOfFeatures=-1,classifiersType = "normal"):
		if executionType == "distributed":
			return self.simulateDistributedClassification(function,classifiersType)
		else:
			return self.computeAccuracyForSingleModel(function,numberOfFeatures)

	#if numberOfFeatures == -1, then compute with all of them
	def computeAccuracyForSingleModel(self,algorithm="SVM",numberOfFeatures=-1):
		if (numberOfFeatures > 0):
			instFeatures = dataPreparation.selectNRandomColumns(self.instancesFeatures,numberOfFeatures)
			#select random numberOfFeatures columns
		else:
			instFeatures = np.array(self.instancesFeatures)

		skf = StratifiedKFold(n_splits=self.kFolds)
		avgScore = 0
		avgF1Macro = 0
		avgF1Micro = 0
		avgF1Weighted = 0
		for train_index, test_index in skf.split(instFeatures, self.instancesClasses):
			resultClasses = classifier.MakeClassification(self.algorithmsIndex[algorithm],instFeatures[train_index],self.instancesClasses[train_index],instFeatures[test_index],"value")
			avgF1Macro += f1_score(self.instancesClasses[test_index], resultClasses, average='macro')
			avgF1Micro += f1_score(self.instancesClasses[test_index], resultClasses, average='micro')
			avgF1Weighted += f1_score(self.instancesClasses[test_index], resultClasses, average='weighted')
			avgScore += accuracy_score(self.instancesClasses[test_index],resultClasses)
		avgScore = avgScore / self.kFolds
		avgF1Macro /= self.kFolds
		avgF1Weighted /= self.kFolds
		avgF1Micro /= self.kFolds
		return avgScore, avgF1Macro, avgF1Micro, avgF1Weighted

	# this function will call all the underlying methods in order to perform data prepation, classification in each simulated agent, and aggregation
	def simulateDistributedClassification(self,combineFunction,classifiersType):
		modelsData = dataPreparation.divideDataSetInPartitions(self.instancesFeatures)
		self.numberOfModels = self.getNumberOfModels(modelsData)
		print "Data loaded!"
		#print modelsData
		outputProbabilities = {}
		skf = StratifiedKFold(n_splits=self.kFolds)
			#X_train, X_test = X[train_index], modelsData[i][test_index]
			#y_train, y_test = y[train_index], instancesClasses[test_index]
		avgScore = 0
		avgF1Macro = 0
		avgF1Micro = 0
		avgF1Weighted = 0
		for train_index, test_index in skf.split(modelsData[0], self.instancesClasses):
			resClass=[0] * self.numberOfModels
			for i in range(self.numberOfModels):
				
				if (combineFunction == "Plurality"):
					resClass[i] = classifier.MakeClassification(i,modelsData[i][train_index],self.instancesClasses[train_index],modelsData[i][test_index],"value",classifiersType)
				else:
					#print("TRAIN:", train_index, "TEST:", test_index)
					vectorProbabilities = classifier.MakeClassification(i,modelsData[i][train_index],self.instancesClasses[train_index],modelsData[i][test_index],"proba",classifiersType)
					outputProbabilities.update( {i : vectorProbabilities } )
			if (combineFunction != "Plurality"):
				rankingsOutput = rankings.makeRankings(outputProbabilities)
				#print rankingsOutput
				estimator = socialChoiceEstimator.socialChoiceEstimator(rankingsOutput)
				resultClasses = estimator.getWinnerClass(combineFunction)
			else:
				resultClasses = [0] * len(resClass[0])
				for i in range(len(resClass[0])):
					tmpList = list()
					for j in range(self.numberOfModels):
						tmpList.append(resClass[j][i])
					resultClasses[i] = max(set(tmpList), key=tmpList.count)
				#resultClasses = 
			#print "Done classification!"
			avgF1Macro += f1_score(self.instancesClasses[test_index], resultClasses, average='macro')
			avgF1Micro += f1_score(self.instancesClasses[test_index], resultClasses, average='micro')
			avgF1Weighted += f1_score(self.instancesClasses[test_index], resultClasses, average='weighted')
			avgScore += accuracy_score(self.instancesClasses[test_index], resultClasses)
		avgScore = avgScore / self.kFolds
		avgF1Macro /= self.kFolds
		avgF1Weighted /= self.kFolds
		avgF1Micro /= self.kFolds
		return avgScore, avgF1Macro, avgF1Micro, avgF1Weighted
		
	def getNumberOfModels(self,data):
		return len(data)