import numpy as np

class KNN(object):
	"""
		Class to cover the implementation of K-Nearest Neighbours.
	"""
	def __init__(self, K):
		""" 
			Constructor for the KNN class. 
			Args:
				K		= The value of K i.e., the number of nearest neighbours to take into consideration
		"""
		super(KNN, self).__init__()
		
		self.K		= K
		
	def fit(self, X, Y):
		"""
			The function to fit the training data.
			Args:
				X		= The training inputs  : numpy.ndarray of shape (n_points, n_params)
				Y		= The training outputs : numpy.ndarray of shape (n_points, )
		"""
		self.X	= X
		self.Y	= Y
	
	def predict(self, X):
		"""
			The function to predict the labels of a batch of testing inputs.
			Args:
				X		= The testing inputs	: numpy.ndarray of shape (n_points, n_params)
			Returns:
				list with predictions in the same order as the inputs
		"""
		predictions	= []

		# For every X in the new_X, find the closest K neighbours from the training Xs
		for i in range(0, X.shape[0]):
			distances 	= []
			for j in range(0, self.X.shape[0]):
				dstnce	= np.linalg.norm(X[i] - self.X[j])
				distances.append(dstnce)
			knns	= np.argpartition(distances, self.K)[0:self.K].tolist()

			# Get the labels of all the closest neighbours and keep count
			count0	= 0
			count1	= 0
			for n in knns:
				if self.Y[n] == 0:
					count0	+= 1
				elif self.Y[n] == 1:
					count1	+= 1
			prdctn	= 1 if count1 > count0 else 0
			predictions.append(prdctn)
			
		return predictions
