from kernels import linear_kernel as lk
from kernels import gaussian_kernel as gk
from kernels import polynomial_kernel as pk
from kernels import linear_gram_matrix as lgm
from kernels import gaussian_gram_matrix as ggm
from kernels import polynomial_gram_matrix as pgm

class MultiKernelfixedrules(object):
	"""
		Class that contains the implementation of Multi Kernel with Fixed Rules
	"""
	def __init__(self, gammas, hyperparameters):
		"""
			Function to initialize certain parameters required for the class
			Args:
				gammas		= The static weights for each class
				hyperparameters	= The values for the "q" and "sigma" in polynomial and Gaussian kernel respectively
		"""
		assert sum(gammas.values()) == 1, "gammas must add up to 1"
		self.gammas		= gammas
		self.hyperparameters	= hyperparameters
		
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
		K	= K + self.gammas['gaussian'] * gk(x_i=x_i, x_j=x_j, sigma=self.hyperparameters['gaussian'])
		K	= K + self.gammas['polynomial'] * pk(x_i=x_i, x_j=x_j, q=self.hyperparameters['polynomial'])
		
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
		
def MultiKernelheuristic(object):
	"""
		Class that contains the implementation of Multi Kernel with Heuristics
	"""
	def __init__(self, hyperparameters, X, Y):
		"""
			Function to initialize certain parameters required for the class
			Args:
				hyperparameters	= The values for the "q" and "sigma" in polynomial and Gaussian kernel respectively
				X		= Inputs		: numpy.ndarray of shape (n_points, n_features)
				Y		= Outputs		: numpy.ndarray of shape (n_points, )
		
		"""
		self.hyperparameters	= hyperparameters

		g_matrices					= {'linear': lgm, 'polynomial': pgm, 'gaussian': ggm}
		eta_linear, eta_polynomial, eta_gaussian	= get_etas(g_matrices, X, Y)
		
		self.gammas		= {'linear': eta_linear, 'gaussian': eta_gaussian, 'polynomial': eta_polynomial}

	def A(g_matrix, y):
		"""
			Function to calculate the value of A
			Args:
				g_matrix		= Gram matrix		: numpy.ndarray of shape (n_points, n_points)
				y			= Output values		: numpy.ndarray of shape (n_points, )
			Returns:
				scalar value : A(K, yyT)
		"""
		from numpy import outer
		ret_val	= g_matrix*outer(y, y).sum()
		ret_val	= ret_val/y.shape[0]
		ret_val	= ret_val/np.sqrt(g_matrix*g_matrix.sum())
		
		return ret_val
		
	def get_etas(g_matrices, X, Y):
		"""
			Function to calculate the eta values
			Args:
				g_matrices		= Dictionary with function objects
				X			= Input matrix		: numpy.ndarray of shape (n_points, n_features)
				Y			= Output matrix		: numpy.ndarray of shape (n_points, )
			Returns:
				3-tuple of etas
		"""
		eta_linear	= A(g_matrices['linear'](X, X), Y)
		eta_polynomial	= A(g_matrices['polynomial'](X, X), Y)
		eta_gaussian	= A(g_matrices['gaussian'](X, X), Y)
		
		eta_linear	= eta_linear/(eta_linear + eta_polynomial + eta_gaussian)
		eta_polynomial	= eta_polynomial/(eta_linear + eta_polynomial + eta_gaussian)
		eta_gaussian	= eta_gaussian/(eta_linear + eta_polynomial + eta_gaussian)
		
		return eta_linear, eta_polynomial, eta_gaussian
		
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
		K	= K + self.gammas['gaussian'] * gk(x_i=x_i, x_j=x_j, sigma=self.hyperparameters['gaussian'])
		K	= K + self.gammas['polynomial'] * pk(x_i=x_i, x_j=x_j, q=self.hyperparameters['polynomial'])
		
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
