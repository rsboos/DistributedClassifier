import math
import copy
import numpy
import pandas
import sklearn


class Data():
	"""Represents the data

	Properties*:
		data -- instances' attributes (pandas.dataframe)
		target -- instances' classes (ndarray)
		classes -- list of classes (ndarray)
		n_columns -- number of columns (int)
		n_lines -- number of lines (int)
	"""

	def __init__(self, filepath, class_column=-1):
		"""Loads the data from a CSV file, normalizes the target and sets the data in the properties

		Keywords arguments:
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

		# Normalizes the target
		preprocess = sklearn.preprocessing.LabelEncoder() # new LabelEncoder
		preprocess.fit(y)								  # inserts the labels

		# Sets all in the properties
		self.__data = x
		self.__target = preprocess.transform(y)	# transforms labels into integers (indexes for classes)
		self.__classes = preprocess.classes_	# list of classes that can be accessed with any target

	@property
	def data(self):
		"""Makes a copy of the data and returns it"""
		return self.__data.copy()

	@data.setter
	def data(self, other):
		"""Sets a new value for data
		
		Keyword arguments:
			other -- the new data (must be convertible to ndarray)
		"""

		self.__data = pandas.DataFrame(__to_ndarray(other))

	@property
	def target(self):
		"""Makes a copy of the target and returns it"""
		return numpy.array(self.__target)

	@target.setter
	def target(self, other):
		"""Sets a new value for target
		
		Keyword arguments:
			other -- the new target (must be convertible to ndarray)
		"""

		self.__target = __to_ndarray(other)

	@property
	def classes(self):
		"""Makes a copy of the classes and returns it"""
		return numpy.array(self.__classes)

	@classes.setter
	def classes(self, other):
		"""Sets a new value for classes
		
		Keyword arguments:
			other -- the new classes (must be convertible to ndarray)
		"""

		self.__classes = __to_ndarray(other)

	@property
	def n_columns(self):
		"""Gets the number of columns and returns it"""
		return self.__data.shape[1] + 1

	@property
	def n_lines(self):
		"""Gets the number of lines from data and returns it"""
		return self.__data.shape[0]

	def __to_ndarray(self, data):
		"""Converts a list or copy a ndarray e returns the result
		
		Keyword arguments:
			data -- data to be converted (must be convertible to ndarray)
		"""

		# If the new data is not a convertible to ndarray...
		if type(data) not in [numpy.ndarray, list]:
			raise TypeError("%s is not a numpy.ndarray. It must be a list or a ndarray." % type(data))

		return numpy.array(data)


class Dataset():
	"""Prepares the data

	Properties:
		trainingset -- Data used by the algorithm for training (Data)
		testset -- Data used for testing the model (Data)
	"""

	def __init__(self, filepath, for_test=0.0, class_column=-1):
		"""Initiates the data and divides into training and testing sets

		Keywords arguments:
			filepath -- file's absolute/relative path
			for_test -- percent of test instances (default 0.0)
			class_column -- number of the class column [0 -> first column, (default -1) -> last column]
		"""

		# Gets the data to split
		self.trainingset = Data(filepath, class_column)

		# Creates a dataset's copy
		self.testset = copy.deepcopy(self.trainingset)

		# Calculates the number of instances for the testing set
		test_ninstances = math.floor(self.trainingset.n_lines * for_test)

		# Split the data
		data = self.trainingset.data 	 # data from one of the copies
		target = self.trainingset.target # targets from one of the copies

		self.trainingset.data = data[test_ninstances:, :]  # gets instances n to len(data)
		self.trainingset.target = target[test_ninstances:] # gets targets n to len(target) 

		self.testset.data = data[:test_ninstances, :]	   # gets instances 0 to n-1
		self.testset.target = target[:test_ninstances]     # gets targets 0 to n-1


