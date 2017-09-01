import numpy
import pandas
import sklearn


class Data():
	"""Represents the data

	Properties*:
	data -- instances' attributes
	target -- instances' classes
	classes -- list of classes

	*All properties are private
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
		x = dataset.ix[:, i:j]
		y = dataset.ix[:, class_column]

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

	@property
	def target(self):
		"""Makes a copy of the target and returns it"""
		return numpy.array(self.__target)

	@property
	def classes(self):
		"""Makes a copy of the classes and returns it"""
		return numpy.array(self.__classes)