from SocialChoiceFunctions import Profile

class socialChoiceEstimator:
	def __init__(self, rankings):
		self.rankings=rankings
		self.data = self.getClassesData()
		self.numberOfClasses = len(self.data)
		self.numberOfModels = len(self.data[0])
		self.numberOfInstances = len(self.data[0][0])
		self.classesScores = None
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

	def prepareDataToPySCF(self,indexClass):
		arr = self.data[indexClass]
		tmp = arr.items()
		ret = []
		for elem in tmp:
			ret.append((1,elem[1]))
		return ret

	def getWinnerClass(self,functionType):
		functionType = "Borda"
		bestClass = [-1] * self.numberOfInstances
		bestScore = [0] * self.numberOfInstances
		for i in range(self.numberOfClasses):
			scores = self.getScoresForClass(functionType,i)
			for j in range(self.numberOfInstances):
				if (bestScore[j] < scores[j]):
					bestScore[j] = scores[j]
					bestClass[j] = i
		return bestClass

	def getScoresForClass(self,functionType,indexClass):
		profileClass = Profile(self.prepareDataToPySCF(indexClass))
		ret = []
		for i in range(self.numberOfInstances):
			ret.append(profileClass.bordaScore(i))
		return ret

"""	def getBordaDecision(self):

		bordaDecision = [-1] * self.numberOfInstances
		maxBordaScore = [0] * self.numberOfInstances
		for indexInstance in range(self.numberOfInstances):
			for indexClass in range(self.numberOfClasses):
				if (bordaScore[])
			indexClass,dataClass in self.data.items():
			maxB
			score[indexClass] = bordaScore(indexClass)


	def getBordaScore():
		
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

	def copelandScore(self,indexClass):
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

"""