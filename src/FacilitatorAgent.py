import dataPreparation
import classifier
import rankings
import socialChoiceEstimator
from sklearn.metrics import accuracy_score

class FacilitatorAgent:
	def __init__(self, dataSetFile):
		self.dataSetFile=dataSetFile
		self.instancesFeatures, self.instancesClasses = dataPreparation.loadDataSetFromFile(self.dataSetFile,"first")
		self.numberOfModels=0
		self.algorithmsIndex = { "SVM": 0, "DecisionTree": 1, "KNN": 2, "NN": 3, "NB": 4, "ECOC": 5}

	def execute(self,executionType="distributed",function=None,numberOfFeatures=-1):
		if executionType == "distributed":
			return self.simulateDistributedClassification(function)
		else:
			return self.computeAccuracyForSingleModel(function,numberOfFeatures)

	#if numberOfFeatures == -1, then compute with all of them
	def computeAccuracyForSingleModel(self,algorithm="SVM",numberOfFeatures=-1):
		if (numberOfFeatures > 0):
			instFeatures = dataPreparation.selectNRandomColumns(self.instancesFeatures,numberOfFeatures)
			#select random numberOfFeatures columns
		
		resultClasses = classifier.MakeClassification(self.algorithmsIndex[algorithm],instFeatures,self.instancesClasses,"value")
		return accuracy_score(self.instancesClasses,resultClasses)

	# this function will call all the underlying methods in order to perform data prepation, classification in each simulated agent, and aggregation
	def simulateDistributedClassification(self,combineFunction):
		modelsData = dataPreparation.divideDataSetInPartitions(self.instancesFeatures)
		self.numberOfModels = self.getNumberOfModels(modelsData)
		
		outputProbabilities = {}
		for i in range(self.numberOfModels):
			vectorProbabilities = classifier.MakeClassification(i,modelsData[i],self.instancesClasses)
			outputProbabilities.update( {i : vectorProbabilities } )
		rankingsOutput = rankings.makeRankings(outputProbabilities)

		estimator = socialChoiceEstimator.socialChoiceEstimator(rankingsOutput)
		results= estimator.getWinnerClass(combineFunction)
		return accuracy_score(self.instancesClasses, results)
		print "Done classification!"

	def getNumberOfModels(self,data):
		return len(data)