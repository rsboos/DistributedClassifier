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
		"""Sets the properties
		
		Keywords arguments:
			x -- instances' attributes (pandas.dataframe)
			y -- instances' classes (ndarray)
		"""

		self.x = x
		self.y = y

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

	def __init__(self, data, n_splits=10, test_size=0.3):
		"""Initiates the data and divides into training and testing sets

		Keyword arguments:
			data -- a Data object
			n_splits -- the number of folds for cross-validation (default 10)
			test_size -- percent of test instances (default 0.3)
		"""

		# Normalizes the target
		preprocess = preprocessing.LabelEncoder() # new LabelEncoder
		preprocess.fit(data.y)				 	  # inserts the labels

		data.y = preprocess.transform(data.y) # transforms labels into integers (indexes for classes)
		self.__classes = preprocess.classes_  # list of classes that can be accessed with any target

		# Splits the data into train and test sets keeping the classes' proportion (stratified)
		train_data, test_data, train_target, test_target = model_selection.train_test_split(data.x, data.y, test_size=test_size, stratify=data.y)

		# Creates the sets for training and testing with the splitted data
		self.trainingset = Data(train_data, train_target)
		self.testset = Data(test_data, test_target)

	@property
	def classes(self):
		"""Makes a copy of the classes and returns it"""
		return numpy.array(self.__classes)

	@property
	def n_features(self):
		"""Gets the number of columns and returns it"""
		return self.__data.shape[1]

	@property
	def n_instances(self):
		"""Gets the number of lines from data and returns it"""
		return self.__data.shape[0]

	@property
	def n_classes(self):
		"""Gets the number of classes from data and returns it"""
		return len(self.__classes)

	@classmethod
	def load(cls, filepath, class_column=-1, n_splits=10, test_size=0.3):
		"""Loads a CSV file, creates a Data object and returns a Dataset object
		
		Keyword arguments:
			filepath -- file's absolute/relative path
			class_column -- number of the class column [0 -> first column, (default -1) -> last column]
			n_splits -- the number of folds for cross-validation (default 10)
			test_size -- percent of test instances (default 0.3)
		"""

		# Loads the data
		data = Data.load_data(filepath, class_column)

		# Creates a Dataset object and returns it
		return cls(data, n_splits, test_size)

	def get_features_distribution(self, n, overlap=0):
		"""Vertically distributes the training data into n parts and returns a list of
		column indexes for each part 

		Keyword arguments:
			n -- number of divisions
			overlap -- If float, should be between 0.0 and 1.0 and represents the percentage
					   of parts' common features. If int, should be less than the number of
					   features and represents the number of common features. If list, 
					   represents the features' columns indexes. By default, the value is set to 0.
		"""

		# Check the type of overlap variable and sets the n_common and common_features, where
		# n_common is the number of common features for each part and
		# common_features is a common features' index's list
		if type(overlap) is int:
			common_features = set(numpy.random.choice(self.n_features, overlap))  # gets a n length list with random numbers between 0-n_features
		elif type(overlap) is float:
			n_common = math.ceil(self.n_features * overlap)						  # calculates the amount of features each part has in common
			common_features = set(numpy.random.choice(self.n_features, n_common)) # gets a n length list with random numbers between 0-n_features
		elif type(overlap) is list:
			common_features = set(overlap)
		else: # is neither int, float and list
			raise TypeError("%s should be a float, an int or a list. %s given." % (str(overlap), type(overlap)))		

		# Calculates the distinct features' indexes
		distinct_features = list(set(range(self.n_features)) - common_features) # common_features' complement

		# Just converts from set to list
		common_features = list(common_features)

		# Permutates the distinct features (randomize)
		distinct_features = numpy.random.permutation(distinct_features)

		# Calculate the number of distinct features per part
		n_distinct = len(distinct_features)

		# Calculates the number of distinc features per part
		n_part_features = math.ceil(n_distinct / n)

		# Empty list for the loop
		distribution = []

		# The main object in the loop is the distinct_features list
		# We want to add a slice of this list with the common_features list
		# for each part
		for i in range(0, n_distinct + 1, n_part_features):
			j = i + n_part_features									# stop index for slice, garantees n features to the part
			part_indexes = common_features + distinct_features[i:j] # list of common and distinct indexes for the part
			distribution.append(part_indexes)						# adds the list to distribution

		# Returns the distribution list
		return distribution