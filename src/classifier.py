from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
import sklearn.svm

def MakeClassification(index,instancesData,classesData,instancesTest,type="proba",classifiersType="normal"):
	classifiers = [
	OneVsRestClassifier(sklearn.svm.SVC(probability=1),4),
	DecisionTreeClassifier(random_state=0),
	KNeighborsClassifier(n_jobs=4),
	MLPClassifier(),
	GaussianNB(),
	OutputCodeClassifier(LinearSVC(random_state=0),code_size=2, random_state=0)
	]
	if (classifiersType == "ova"):
		classifiers = [
			OneVsRestClassifier(sklearn.svm.SVC(probability=1),4),
			OneVsRestClassifier(DecisionTreeClassifier(random_state=0),4),
			OneVsRestClassifier(KNeighborsClassifier(),4),
			OneVsRestClassifier(MLPClassifier(),4),
			OneVsRestClassifier(sklearn.svm.SVC(probability=1),4)
		]
	if (index >= len(classifiers)):
		print "ERROR. The index is not valid."
		return None
	else:
		#print "Performing classification"
		if type == "proba":
			return classifiers[index].fit(instancesData,classesData).predict_proba(instancesTest)
		else:
			return classifiers[index].fit(instancesData,classesData).predict(instancesTest)