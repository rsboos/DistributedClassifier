import sys
from pathlib import Path
import FacilitatorAgent

import os

def createDir(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)

class Executer:
	#algorithmType = {distributed,local}
	# socialChoiceFunction = { Borda, Copeland, Dowdall, Plurality }
	def __init__(self, dataSetFile,classesPlace,algorithmType,function,classifiersType,kFoldNumber,subsetFeatures,totalFilename,resultsFilename):
		self.dataSetFile = dataSetFile
		self.kFoldNumber = kFoldNumber
		self.algorithmType = algorithmType
		self.function = function
		self.subsetFeatures = subsetFeatures
		self.classesPlace = classesPlace
		self.classifiersType = classifiersType
		self.totalFilename = "../results/"+totalFilename
		self.resultsFilename = "../results/"+resultsFilename

	def getAverageAccuracy(self):
		avgScore, avgF1Macro, avgF1Micro, avgF1Weighted = (0,0,0,0)
		Executer = FacilitatorAgent.FacilitatorAgent(self.dataSetFile,self.classesPlace,self.kFoldNumber,self.totalFilename)
		for i in range(10):
			ret = Executer.execute(self.algorithmType,self.function,self.subsetFeatures,self.classifiersType)
			avgScore += ret[0]
			avgF1Macro += ret[1]
			avgF1Micro += ret[2]
			avgF1Weighted += ret[3]
		avgScore /= 10
		avgF1Macro /= 10
		avgF1Micro /= 10
		avgF1Weighted /= 10
		with open(self.resultsFilename, "a") as myfile:
			myfile.write(str(avgF1Weighted)+"\t"+str(avgF1Micro)+"\t"+str(avgF1Macro)+"\t"+str(avgScore)+"\n")
		return avgScore, avgF1Macro, avgF1Micro, avgF1Weighted
	
def main(argv):
	if (len(argv) < 3):
		print "Execution should be in the format: python main.py dataset.txt classesPlace(first or last) algorithmType ( distributed or local) function (combine function or classify algorithm) classifiersType (optional => normal or ova) kFoldNumber(optional => default = 10) isSmallLocalModel (optional = 0 or 1)"
	else:
		dataSetFile = argv[0]
		classesPlace = argv[1]
		algorithm = argv[2]
		function = argv[3]
		if (len(argv) > 4):
			classifiersType = argv[4]
		else:
			classifiersType = "normal"
		if (len(argv) > 5):
			kFoldNumber = int(argv[5])
		else:
			kFoldNumber = 10
		if (len(argv) > 6):
			subsetFeatures = int(argv[6])
		else:
			subsetFeatures = 0
		file = Path(dataSetFile)
		if file.is_file():
			strLocalModel=""
			rawName = os.path.basename(dataSetFile)[:-8]
			if (subsetFeatures == 0):
				strLocalModel=rawName+"/localComplete/"
			else:
				strLocalModel=rawName+"/localSmall/"
			createDir("../results/"+strLocalModel)
			totalFilename = strLocalModel+algorithm+"_"+function+"_"+classifiersType+"_all_"+os.path.basename(dataSetFile)
			resultsFilename = strLocalModel+algorithm+"_"+function+"_"+classifiersType+"_results_"+os.path.basename(dataSetFile)
			flow = Executer(dataSetFile,classesPlace,algorithm,function,classifiersType,kFoldNumber,subsetFeatures,totalFilename,resultsFilename)
			flow.getAverageAccuracy()
		else:
			print "The dataset file should have a valid path"

if __name__ == "__main__":
	main(sys.argv[1:])