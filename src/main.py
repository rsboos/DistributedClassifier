import sys
from pathlib import Path
import FacilitatorAgent

class Executer:
	#algorithmType = {distributed,local}
	# socialChoiceFunction = { Borda, Copeland, Dowdall, Plurality }
	def __init__(self, dataSetFile,classesPlace,algorithmType,function,classifiersType = "normal",kFoldNumber=10,subsetFeatures=-1):
		self.dataSetFile = dataSetFile
		self.kFoldNumber = kFoldNumber
		self.algorithmType = algorithmType
		self.function = function
		self.subsetFeatures = subsetFeatures
		self.classesPlace = classesPlace
		self.classifiersType = classifiersType 

	def getAverageAccuracy(self):
		Executer = FacilitatorAgent.FacilitatorAgent(self.dataSetFile,self.classesPlace,self.kFoldNumber)
		return Executer.execute(self.algorithmType,self.function,self.subsetFeatures,self.classifiersType)
	
def main(argv):
	if (len(argv) < 3):
		print "Execution should be in the format: python main.py dataset.txt classesPlace(first or last) algorithmType ( distributed or local) function (combine function or classify algorithm) classifiersType (optional => normal or ova) kFoldNumber(optional => default = 10) subsetFeatures (optional)"
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
			subsetFeatures = -1
		file = Path(dataSetFile)
		if file.is_file():
			flow = Executer(dataSetFile,classesPlace,algorithm,function,classifiersType,kFoldNumber,subsetFeatures)
			print flow.getAverageAccuracy()
		else:
			print "The dataset file should have a valid path"

if __name__ == "__main__":
	main(sys.argv[1:])