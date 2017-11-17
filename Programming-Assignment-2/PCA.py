import numpy as np

class PCA(object):
    W = None
    def __init__(self, n_components):
        """
            PCA object constructor for Principle Components Analysis
            Args:
                n_components    : Int, number of principal components required
        """
        super(PCA, self).__init__()

        self.k = n_components
        print("PCA object: n_components: {0}".format(self.k))

    def fit(self, X):
        """
            PCA fit method
            Args:
                X               : numpy.ndarray of shape (n_points, n_features)
        """
        assert self.k <= X.shape[1], "Number of components cannot be greater than dimensionality of input"
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # Compute Covariance matrix for features
        cov_X = np.cov(X, rowvar=False)

        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eig(cov_X)

        # Get a descending order of eigenvalues to choose k eigenvectors
        order = np.argsort(eigvals)[::-1]
        self.W = eigvecs[:,order[:self.k]]

    def transform(self, X):
        """
            PCA transform method
            Args:
                X               : numpy.ndarray of shape (n_points, n_features)
            Returns:
                Transformed X with k Principal components
        """
        mean = np.mean(X, axis=0)
        X = X - mean

        # Transform based on W found by fitting
        return np.dot(self.W.T, X.T).T
