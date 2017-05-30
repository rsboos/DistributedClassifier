import sys
import dataPreparation
import classifier
import rankings
import socialChoiceEstimator



class FacilitatorAgent:
	def __init__(self, dataSetFile):
		self.dataSetFile=dataSetFile
		self.numberOfModels=0

	# this function will call all the underlying methods in order to perform data prepation, classification in each simulated agent, and aggregation
	def executeAlgorithm(self,debug=False):
		instancesFeatures, instancesClasses = dataPreparation.loadDataSetFromFile(self.dataSetFile)
		modelsData = dataPreparation.divideDataSetInPartitions(instancesFeatures)
		self.numberOfModels = self.getNumberOfModels(modelsData)
		
		outputProbabilities = {}
		for i in range(self.numberOfModels):
			vectorProbabilities = classifier.MakeClassification(i,modelsData[i],instancesClasses)
			outputProbabilities.update( {i : vectorProbabilities } )
		rankingsOutput = rankings.makeRankings(outputProbabilities) 
		#rankings = { i (modelIndex: { j (classIndex): preference } }

		estimator = socialChoiceEstimator.socialChoiceEstimator(rankingsOutput)
		print estimator.getWinnerClass("BordaCount")
		

	def getNumberOfModels(self,data):
		return len(data)
	
def main(argv):
    Executer = FacilitatorAgent("../datasets/iris.data.txt")
    Executer.executeAlgorithm()

if __name__ == "__main__":
    main(sys.argv[1:])