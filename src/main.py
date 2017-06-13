import sys
from pathlib import Path
import FacilitatorAgent

class Executer:
	#algorithmType = {distributed,local}
	# socialChoiceFunction = { Borda, Copeland, Dowdall, Plurality }
	def __init__(self, dataSetFile,classesPlace,algorithmType,function,kFoldNumber=10,subsetFeatures=-1):
		self.dataSetFile = dataSetFile
		self.kFoldNumber = kFoldNumber
		self.algorithmType = algorithmType
		self.function = function
		self.subsetFeatures = subsetFeatures
		self.classesPlace = classesPlace

	def getAverageAccuracy(self):
		Executer = FacilitatorAgent.FacilitatorAgent(self.dataSetFile,self.classesPlace,self.kFoldNumber)
		return Executer.execute(self.algorithmType,self.function,self.subsetFeatures)
	
def main(argv):
	if (len(argv) < 3):
		print "Execution should be in the format: python main.py dataset.txt classesPlace(first or last) algorithmType ( distributed or local) function (combine function or classify algorithm) kFoldNumber(optional => default = 10) subsetFeatures (optional)"
	else:
		dataSetFile = argv[0]
		classesPlace = argv[1]
		algorithm = argv[2]
		function = argv[3]
		if (len(argv) > 4):
			kFoldNumber = int(argv[4])
		else:
			kFoldNumber = 10
		if (len(argv) > 5):
			subsetFeatures = int(argv[5])
		else:
			subsetFeatures = -1
		file = Path(dataSetFile)
		if file.is_file():
			flow = Executer(dataSetFile,classesPlace,algorithm,function,kFoldNumber,subsetFeatures)
			print flow.getAverageAccuracy()
		else:
			print "The dataset file should have a valid path"

if __name__ == "__main__":
	main(sys.argv[1:])