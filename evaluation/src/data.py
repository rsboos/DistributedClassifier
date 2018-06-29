from pandas import read_csv
from sklearn.preprocessing import LabelEncoder


class Data():
    """Represent the data.

	Properties*:
		x -- instances' attributes (ndarray)
		y -- instances' classes (ndarray)
	"""

    def __init__(self, x, y):
        self.x = x
        self.__discretize(y)

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
        return self.classes.size

    @classmethod
    def load(cls, filepath, class_column=-1):
        """Load a headless CSV file and return a Data object.

		Keyword arguments:
			filepath -- file's absolute/relative path
			class_column -- number of the class column [0 -> first column, (default -1) -> last column]
		"""
        has_header = cls.__has_header(filepath)
        dataset = read_csv(filepath, header=has_header)
        dataset = dataset.values

        n_columns = dataset.shape[1]

        i = class_column + 1
        j = n_columns + class_column

        x = dataset[:, i:j]
        y = dataset[:, class_column]

        return cls(x, y)

    @staticmethod
    def __has_header(filepath):
        file = open(filepath, 'r')
        line = file.readline()
        file.close()

        try:
            line = [float(n) for n in line.split(',')]
            return None
        except ValueError:
            return 0

    def __discretize(self, y):
        encoder = LabelEncoder()
        self.y = encoder.fit_transform(y)
        self.classes = encoder.classes_
