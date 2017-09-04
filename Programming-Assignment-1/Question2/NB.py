import numpy as np

class NB(object):
	"""
		Class to cover the implementation of Naive Bayes with a Gaussian likelihood
	"""
	def __init__(self):
		"""
			Constructor for the NB class.
		"""
		super(NB, self).__init__()
		
	def fit(self, X, Y):
		"""
			The function to fit the training data.
			Args:
				X		= The training inputs 	: numpy.ndarray of shape (n_points, n_features)
				Y		= The training outputs	: numpy.ndarray of shape (n_points, )
		"""
		assert X.shape[0] == Y.shape[0], "Unmatched number of datapoints"
		
		# Get the count of points in each class and class probabilities (prior)
		n_class	= np.array([0, 0])
		Y	= Y.astype(int)
		for y in Y:
			n_class[y] += 1
		assert sum(n_class) == Y.shape[0], "Non-binary classification data"
					
		self.p_class		= [n/sum(n_class) for n in n_class]
		
		# Get the means of the features for separate classes
		means	= np.array([np.full(X.shape[1], 0), np.full(X.shape[1], 0)]).astype(np.float64)
		stddevs	= means.copy()
		
		for i in range(0, X.shape[0]):
			means[Y[i]] += X[i]/n_class[Y[i]]
		
		for i in range(0, X.shape[0]):
			stddevs[Y[i]] += np.power((X[i] - Y[i]), 2)/n_class[Y[i]]
		stddevs	= np.power(stddevs, 0.5)
		
		self.means	= means
		self.stddevs	= stddevs
		
	def predict(self, X):
		"""
			The function to predict the labels of a batch of testing inputs.
			Args:
				X		= The testing inputs	: numpy.ndarray of shape (n_points, n_features)
			Returns:
				list with prediction in the same order as the inputs
		"""
		predictions	= []
		for i in range(0, X.shape[0]):
			x	= X[i]
			probs	= []
			for j in range(0, len(self.p_class)):
				p	= p_class[j]
				nb_p	= gaussian_prob(x=x, means=self.means[j], stddevs=self.stddevs[j])
				p	= p*np.prod(nb_p)
				probs.append(p)
			predictions.append(np.argmax(probs))
		return predictions
		
def gaussian_prob(x, means, stddevs):
	pass_	= np.power(x - means, 2)
	pass_	= -pass_/(2*np.power(stddevs, 2))
	pass_	= np.exp(pass_)
	pass_	= pass_/(np.sqrt(2*np.pi)*stddevs)
	return pass_
