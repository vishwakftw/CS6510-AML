import numpy as np

def linear_kernel(x_i, x_j):
	return np.dot(x_i, x_j)
	
def polynomial_kernel(x_i, x_j, q):
	assert type(q) == int, "q is supposed to be an integer"
	
	return np.power(np.dot(x_i, x_j) + 1, q)
	
def gaussian_kernel(x_i, x_j, sigma):
	assert sigma > 0, "sigma is supposed to be strictly positive"
	
	k	= x_i.flatten() - x_j.flatten()
	k	= -np.power(k, 2).sum()
	k	= np.exp(k/sigma**2)
	return k
	
def linear_gram_matrix(X, Y):
	assert X.shape[1] == Y.shape[1], "Cannot perform dot product on vectors of dimensions {0} and {1}".format(X.shape[1], Y.shape[1])
	gram_matrix	= np.empty(shape=(X.shape[0], Y.shape[0]))
	for i, x in enumerate(X):
		for j, y in enumerate(Y):
			gram_matrix[i, j]	= linear_kernel(x, y)
			
	return gram_matrix
	
def polynomial_gram_matrix(X, Y, q):
	assert X.shape[1] == Y.shape[1], "Cannot perform dot product on vectors of dimensions {0} and {1}".format(X.shape[1], Y.shape[1])
	gram_matrix	= np.empty(shape=(X.shape[0], Y.shape[0]))
	for i, x in enumerate(X):
		for j, y in enumerate(Y):
			gram_matrix[i, j]	= polynomial_kernel(x, y, q)
			
	return gram_matrix
	
def gaussian_gram_matrix(X, Y, sigma):
	assert X.shape[1] == Y.shape[1], "Cannot perform element-wise subtraction on vectors of dimensions {0} and {1}".format(X.shape[1], Y.shape[1])
	gram_matrix	= np.empty(shape=(X.shape[0], Y.shape[0]))
	for i, x in enumerate(X):
		for j, y in enumerate(Y):
			gram_matrix[i, j]	= gaussian_kernel(x, y, sigma)
