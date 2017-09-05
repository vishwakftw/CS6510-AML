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
		
		# Maintaining classwise split for quicker mean and variance calculation
		classwise_split	= []
		for i in range(0, len(n_class)):
			classwise_split.append(np.empty(shape=(n_class[i], X.shape[1])))
			
		for i in range(0, X.shape[0]):
			n_class[Y[i]]	-= 1
			classwise_split[Y[i]][n_class[Y[i]]]	= X[i]
		
		assert	n_class.all() == 0, "Mis-calculation in class frequencies"

		# Original data will be discarded (which came as argument)
		del X
		
		means	= []
		vrncs	= []
		for i in range(0, len(n_class)):
			means.append(np.mean(classwise_split[i], axis=0))
			vrncs.append(np.var(classwise_split[i], axis=0))

		# Classwise split will be discarded as well
		del classwise_split
						
		self.means	= np.array(means)
		self.vars	= np.array(vrncs)
		
	def predict(self, X):
		"""
			The function to predict the labels of a batch of testing inputs.
			Args:
				X		= The testing inputs	: numpy.ndarray of shape (n_points, n_features)
			Returns:
				list with prediction in the same order as the inputs
		"""
		predictions	= []
		probs	= []

		for i in range(0, len(self.p_class)):
			prior	= np.log(self.p_class[i])
			llhood	= log_gaussian_prob(x=X, means=self.means[i], variances=self.vars[i])
			probs.append((prior + llhood).sum(axis=1))

		predictions	= np.argmax(np.array(probs), axis=0).tolist()
		return predictions, probs
		
def log_gaussian_prob(x, means, variances):

	# To prevent runtime overflow, we add a very small value
	# We are not calculating exact gaussian probability, instead the log of the expression with variables (means and variances)

	variances	+= 1e-12
	pass_	= ((x - means)**2)/variances
	pass_	= -pass_ - 0.5*np.log(variances)
	
	return pass_
