import math
import copy
import numpy
import pandas

from sklearn import preprocessing, model_selection


class Data():
	"""Represents the data

	Properties*:
		x -- instances' attributes (pandas.dataframe)
		y -- instances' classes (ndarray)
	"""

	def __init__(self, x, y):
		"""Nomalize the data and sets the properties

		Keywords arguments:
			x -- instances' attributes (pandas.dataframe)
			y -- instances' classes (ndarray)
		"""

		# Normalizes the target
		preprocess = preprocessing.LabelEncoder() # new LabelEncoder
		preprocess.fit(y)				 	      # inserts the labels

		self.x = x						   # sets the instances
		self.y = preprocess.transform(y)   # transforms labels into integers (indexes for classes)
		self.classes = preprocess.classes_ # list of classes that can be accessed with any target

	@property
	def n_features(self):
		"""Gets the number of columns and returns it"""
		return self.x.shape[1]

	@property
	def n_instances(self):
		"""Gets the number of lines from data and returns it"""
		return self.x.shape[0]

	@property
	def n_classes(self):
		"""Gets the number of classes from data and returns it"""
		return len(self.classes)

	@classmethod
	def load(cls, filepath, class_column=-1):
		"""Loads a CSV file and returns a tuple with instances' features and classes (x,y)

		Keyword arguments:
			filepath -- file's absolute/relative path
			class_column -- number of the class column [0 -> first column, (default -1) -> last column]
		"""

		# Loads the CSV file
		dataset = pandas.read_csv(filepath, header=None)

		# Gets the number of columns: dataset.shape -> (#lines, #columns)
		n_columns = dataset.shape[1]

		# Initial and final column index (exclusive on stop)
		i = class_column + 1			# initial
		j = n_columns + class_column	# final + 1 (because it's exclusive)

		# Separates the data from the classes (target)
		x = dataset.ix[:, i:j]			# all lines and all columns (except the class column)
		y = dataset.ix[:, class_column] # all lines and the class column

		# Creates a Data object and returns it
		return cls(x, y)

	def map_classes(self, predictions):
		"""Map predictions indexes to classes.

		Keyword argument:
			predictions -- a list of predictions
		"""
		pred = map(lambda x: self.classes[x], predictions)
		return list(pred)
