from kernels import linear_kernel as lk
from kernels import gaussian_kernel as gk
from kernels import polynomial_kernel as pk

class MultiKernelFR(object):
	"""
		Class that contains the implementation of Multi Kernel with Fixed Rules
	"""
	def __init__(self, gammas, hyperparameters):
		"""
			Function to initialize certain parameters required for the class
			Args:
				gammas		= The static weights for each class
				hyperparams	= The values for the "q" and "sigma" in polynomial and Gaussian kernel respectively
		"""
		super(MultiKernelfixedrules, self).__init__()
		self.gammas	= gammas
		self.hyparams	= hyperparameters
		
	def __call__(self, x_i, x_j):
		"""
			Function to calculate the kernel based on the inputs
			Args:
				x_i		= Input1	: numpy.ndarray of shape (n_features, )
				x_j		= Input2	: numpy.ndarray of shape (n_features, )
			Returns:
				K(X, Y)
		"""
		K	= 0

		K	= K + self.gammas['linear'] * lk(x_i=x_i, x_j=x_j)
		K	= K + self.gammas['gaussian'] * gk(x_i=x_i, x_j=x_j, sigma=self.hyparams['gaussian'])
		K	= K + self.gammas['polynomial'] * pk(x_i=x_i, x_j=x_j, q=self.hyparams['polynomial'])
		
		return K
		
	def gram_matrix(self, X, Y):
		"""
			Function to calculate the gram matrix (matrix of kernels) based on the inputs
			Args:
				X		= Input1 		: numpy.ndarray of shape (n_points_X, n_features)
				Y		= Input2		: numpy.ndarray of shape (n_points_Y, n_features)
			Returns:
				G(X, Y)		= Matrix of K(x, y)	: numpy.ndarray of shape (n_points_X, n_points_Y)
		"""
		from numpy import empty
		assert X.shape[1] == Y.shape[1], "Cannot perform dot product or element-wise subtraction on vectors of dimensions {0} and {1}".format(X.shape[1], Y.shape[1])
		grm_mtrx	= empty(shape=(X.shape[0], Y.shape[0]))
		for i, x in enumerate(X):
			for j, y in enumerate(Y):
				grm_mtrx[i, j]	= self.__call__(x, y)
				
		return grm_mtrx
