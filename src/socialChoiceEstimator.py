
class socialChoiceEstimator:
	def __init__(self, rankings):
		self.rankings=rankings
		self.data = self.getClassesData()
		#print self.data

	def getClassesData(self): # transform model,class,data into class,model,data
		ret = {}
		for index, classes in self.rankings.items():
			for indexClass,ranking in classes.items():
				if indexClass in ret:
					ret[indexClass].update( {index : ranking} )
				else:
					ret[indexClass] = {}
					ret[indexClass].update( {index : ranking} )
		return ret

	def bordaScore(self,indexClass):
		arr = data[indexClass]
		numberOfModels = len(arr)
		numberOfInstances = len(arr[0])
		scores = [0] * numberOfInstances
		for ranking in arr:
			curIndex=numberOfInstances-1
			for element in ranking:
				scores[element]+=curIndex
				curIndex-=1
		return scores

