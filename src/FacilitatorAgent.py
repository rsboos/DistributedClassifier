import dataPreparation
import classifier
import rankings
import socialChoiceEstimator
from sklearn.metrics import accuracy_score

class FacilitatorAgent:
	def __init__(self, dataSetFile):
		self.dataSetFile=dataSetFile
		self.numberOfModels=0

	#if numberOfFeatures == -1, then compute with all of them
	def computeAccuracyForSingleModel(self,algorithm="SVM",numberOfFeatures=-1):
		print "hey"

	# this function will call all the underlying methods in order to perform data prepation, classification in each simulated agent, and aggregation
	def simulateDistributedClassification(self,combineFunction="Borda",socialdebug=False):
		instancesFeatures, instancesClasses = dataPreparation.loadDataSetFromFile(self.dataSetFile)
		modelsData = dataPreparation.divideDataSetInPartitions(instancesFeatures)
		self.numberOfModels = self.getNumberOfModels(modelsData)
		
		outputProbabilities = {}
		for i in range(self.numberOfModels):
			vectorProbabilities = classifier.MakeClassification(i,modelsData[i],instancesClasses)
			outputProbabilities.update( {i : vectorProbabilities } )
		rankingsOutput = rankings.makeRankings(outputProbabilities)

		estimator = socialChoiceEstimator.socialChoiceEstimator(rankingsOutput)
		results= estimator.getWinnerClass(combineFunction)
		return accuracy_score(instancesClasses, results)

	def getNumberOfModels(self,data):
		return len(data)