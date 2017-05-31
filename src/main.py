import sys
from pathlib import Path
import FacilitatorAgent


	
def main(argv):
	if (len(argv) != 1):
		print "Execution should be in the format: python main.py dataset.txt"
	else:
		dataSetFile = argv[0]
		file = Path(dataSetFile)
		if file.is_file():
			Executer = FacilitatorAgent.FacilitatorAgent(dataSetFile)
			print Executer.simulateDistributedClassification()
		else:
			print "The dataset file should have a valid path"

if __name__ == "__main__":
	main(sys.argv[1:])