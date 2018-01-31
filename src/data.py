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
		dataset = pandas.read_csv(filepath)

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


class Dataset():
	"""Prepares the data

	Properties:
		trainingset -- Data used by the algorithm for training (Data)
		testset -- Data used for testing the model (Data)
		classes -- list of classes (ndarray)
		n_features -- number of features (int)
		n_instances -- number of instances (int)
		n_classes -- number of classes (int)
	"""

	def __init__(self, train, test=None):
		"""Sets the properties

		Keyword arguments:
			train -- a Data object with training data
			test -- a Data object with test data (default None)
		"""

		# Creates the sets for training and testing with the splitted data
		self.trainingset = train
		self.testset = test

	@classmethod
	def load(cls, filepath, class_column=-1, test_size=0):
		"""Loads a CSV file, creates a Data object and returns a Dataset object

		Keyword arguments:
			filepath -- file's absolute/relative path
			class_column -- number of the class column [0 -> first column, (default -1) -> last column]
			test_size -- percent of test instances (default 0.3)
		"""

		# Loads the data
		data = Data.load(filepath, class_column)

		# Splits into training and test sets
		train, test = cls.train_test_split(data, test_size)

		# Creates a Dataset object and returns it
		return cls(train, test)

	@staticmethod
	def train_test_split(data, test_size=0):
		"""Splits the data into train and test sets keeping the classes'
		proportion (stratified) and returns two data objects (train, test)

		Keyword arguments:
			data -- a Data object not splitted
			test_size -- percent of test instances (default 0.3)
		"""

		# Splits
		train_data, test_data, train_target, test_target = model_selection.train_test_split(data.x, data.y, test_size=test_size, stratify=data.y)

		train = Data(train_data, train_target)  # creates a Data object for training
		test = Data(test_data, test_target)		# creates a Data object for testing

		# Returns the Data objects
		return train, test
