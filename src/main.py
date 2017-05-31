import sys
from pathlib import Path
import FacilitatorAgent

class Executer:
	#algorithmType = {distributed,local}
	# socialChoiceFunction = { Borda, Copeland, Dowdall, Plurality }
	def __init__(self, dataSetFile,algorithmType,function,kFoldNumber=10,subsetFeatures=-1):
		self.dataSetFile = dataSetFile
		self.kFoldNumber = kFoldNumber
		self.algorithmType = algorithmType
		self.function = function
		self.subsetFeatures = subsetFeatures

	def getAverageAccuracy(self):
		if self.algorithmType == "local" and self.subsetFeatures == -1:
			Executer = FacilitatorAgent.FacilitatorAgent(self.dataSetFile)
			return Executer.execute(self.algorithmType,self.function,self.subsetFeatures)
		else:
			curAccuracy = 0
			for i in range(self.kFoldNumber):
				Executer = FacilitatorAgent.FacilitatorAgent(self.dataSetFile)
				curAccuracy+=Executer.execute(self.algorithmType,self.function,self.subsetFeatures)
			return curAccuracy/self.kFoldNumber

	
def main(argv):
	if (len(argv) < 3):
		print "Execution should be in the format: python main.py dataset.txt algorithmType ( distributed or local) function (combine function or classify algorithm) kFoldNumber(optional => default = 10) subsetFeatures (optional)"
	else:
		dataSetFile = argv[0]
		algorithm = argv[1]
		function = argv[2]
		if (len(argv) > 3):
			kFoldNumber = int(argv[3])
		else:
			kFoldNumber = 10
		if (len(argv) > 4):
			subsetFeatures = int(argv[4])
		else:
			subsetFeatures = -1
		file = Path(dataSetFile)
		if file.is_file():
			flow = Executer(dataSetFile,algorithm,function,kFoldNumber,subsetFeatures)
			print flow.getAverageAccuracy()
		else:
			print "The dataset file should have a valid path"

if __name__ == "__main__":
	main(sys.argv[1:])