from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import sklearn.svm

def MakeClassification(index,instancesData,classesData):
	classifiers = [
	OneVsRestClassifier(sklearn.svm.SVC(probability=1)),
	DecisionTreeClassifier(random_state=0),
	KNeighborsClassifier(),
	MLPClassifier(alpha=1),
	GaussianNB()
	]
	if (index >= len(classifiers)):
		print "ERROR. The index is not valid."
		return None
	else:
		return classifiers[index].fit(instancesData,classesData).predict_proba(instancesData)