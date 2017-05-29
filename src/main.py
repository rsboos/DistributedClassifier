
import sys
import dataPreparation
import classifier




"""
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
"""
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
		
		
		#makeRankings(outputProbabilities,numberOfModels)
		#agregateRankings()

	def getNumberOfModels(self,data):
		return len(data)
	
def main(argv):
    Executer = FacilitatorAgent("../datasets/iris.data.txt")
    Debug=True
    Executer.executeAlgorithm(Debug)

if __name__ == "__main__":
    main(sys.argv[1:])