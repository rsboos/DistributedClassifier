from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import f1_score

class benchMarks:
	def __init__(self, truth, pred):
		self.pred = pred
		self.truth = truth

	def getF1Scores(self):
		return f1_score(self.instancesClasses, resultClasses, average='macro'),f1_score(self.instancesClasses, resultClasses, average='micro'),f1_score(self.instancesClasses, resultClasses, average='weighted')

	def multiclass_roc_auc_score(self, average="macro"):
		lb = LabelBinarizer()
		lb.fit(self.truth)
		truth = lb.transform(self.truth)
		pred = lb.transform(self.pred)
		return roc_auc_score(truth, pred, average=average)