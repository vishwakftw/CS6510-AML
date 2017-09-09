import numpy as np

def linear_kernel(x_i, x_j):
	"""
		Function to calculate linear kernel
		Args:
			x_i	= Input1	: numpy.ndarray of shape (n_features, )
			x_j	= Input2	: numpy.ndarray of shape (n_features, )
		Returns:
			linear kernel of x_i and x_j
	"""
	return np.dot(x_i, x_j)
	
def polynomial_kernel(x_i, x_j, q):
	"""
		Function to calculate polynomial kernel
		Args:
			x_i	= Input1	: numpy.ndarray of shape (n_features, )
			x_j	= Input2	: numpy.ndarray of shape (n_features, )
			q	= Degree	: positive integer
		Returns:
			polynomial kernel of x_i and x_j
	"""
	assert type(q) == int, "q is supposed to be an integer"
	
	return np.power(np.dot(x_i, x_j) + 1, q)
	
def gaussian_kernel(x_i, x_j, sigma):
	"""
		Function to calculate Gaussian kernel
		Args:
			x_i	= Input1	: numpy.ndarray of shape (n_features, )
			x_j	= Input2	: numpy.ndarray of shape (n_features, )
			sigma	= Hyperparameter: positive number
		Returns:
			Gaussian kernel of x_i and x_j
	"""
	assert sigma > 0, "sigma is supposed to be strictly positive"
	
	k	= np.linalg.norm(x_i - x_j)/sigma
	k	= np.exp(-(k**2))
	return k
	
def linear_gram_matrix(X, Y):
	"""
		Function to calculate the linear gram matrix
		Args:
			X	= Input1	: numpy.ndarray of shape (n_points_X, n_features)
			Y	= Input2	: numpy.ndarray of shape (n_points_Y, n_features)
		Returns:
			Linear Gram Matrix of X and Y	: numpy.ndarray of shape (n_points_X, n_points_Y)
	"""
	assert X.shape[1] == Y.shape[1], "Cannot perform dot product on vectors of dimensions {0} and {1}".format(X.shape[1], Y.shape[1])
	gram_matrix	= np.empty(shape=(X.shape[0], Y.shape[0]))
	if X.shape[0] == Y.shape[0]:
		for i in range(0, X.shape[0]):
			for j in range(0, X.shape[0] - i):
				gram_matrix[i, j]	= gram_matrix[j, i]	= linear_kernel(X[i], Y[j])

	else:
		for i, x in enumerate(X):
			for j, y in enumerate(Y):
				gram_matrix[i, j]	= linear_kernel(x, y)
			
	return gram_matrix
	
def polynomial_gram_matrix(X, Y, q):
	"""
		Function to calculate the polynomial gram matrix
		Args:
			X	= Input1	: numpy.ndarray of shape (n_points_X, n_features)
			Y	= Input2	: numpy.ndarray of shape (n_points_Y, n_features)
			q	= Degree	: positive integer
		Returns:
			Polynomial Gram Matrix of X and Y	: numpy.ndarray of shape (n_points_X, n_points_Y)
	"""
	assert X.shape[1] == Y.shape[1], "Cannot perform dot product on vectors of dimensions {0} and {1}".format(X.shape[1], Y.shape[1])
	gram_matrix	= np.empty(shape=(X.shape[0], Y.shape[0]))
	if X.shape[0] == Y.shape[0]:
		for i in range(0, X.shape[0]):
			for j in range(0, X.shape[0] - i):
				gram_matrix[i, j]	= gram_matrix[j, i]	= polynomial_kernel(X[i], Y[j], q)

	else:
		for i, x in enumerate(X):
			for j, y in enumerate(Y):
				gram_matrix[i, j]	= polynomial_kernel(x, y, q)
			
	return gram_matrix
				
def gaussian_gram_matrix(X, Y, sigma):
	"""
		Function to calculate the Gaussian gram matrix
		Args:
			X	= Input1	: numpy.ndarray of shape (n_points_X, n_features)
			Y	= Input2	: numpy.ndarray of shape (n_points_Y, n_features)
			sigma	= hyperparameter: positive number
		Returns:
			Gaussian Gram Matrix of X and Y	: numpy.ndarray of shape (n_points_X, n_points_Y)
	"""
	assert X.shape[1] == Y.shape[1], "Cannot perform element-wise subtraction on vectors of dimensions {0} and {1}".format(X.shape[1], Y.shape[1])
	gram_matrix	= np.empty(shape=(X.shape[0], Y.shape[0]))
	if X.shape[0] == Y.shape[0]:
		for i in range(0, X.shape[0]):
			for j in range(i, X.shape[0]):
				gram_matrix[i, j]	= gram_matrix[j, i]	= gaussian_kernel(X[i], Y[j], sigma)

	else:
		for i, x in enumerate(X):
			for j, y in enumerate(Y):
				gram_matrix[i, j]	= gaussian_kernel(x, y, sigma)
			
	return gram_matrix
