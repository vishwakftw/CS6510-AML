import numpy as np

class NB(object):
	"""
		Class to cover the implementation of Naive Bayes with a Gaussian likelihood
	"""
	def __init__(self):
		"""
			Constructor for the NB class.
		"""
		super(NB, self).__init__
		
	def fit(self, X, Y):
		"""
			The function to fit the training data.
			Args:
				X		= The training inputs 	: numpy.ndarray of shape (n_points, n_features)
				Y		= The training outputs	: numpy.ndarray of shape (n_points, )
		"""
		assert X.shape[0] == Y.shape[0], "Unmatched number of datapoints"
		
		# Get the count of points in each class and class probabilities (prior)
		n_class	= [0, 0]
		for y in Y:
			n_class[y] += 1
		assert sum(n_class) == Y.shape[0], "Non-binary classification data"
					
		p_class		= [n/sum(n_class) for n in n_class]
