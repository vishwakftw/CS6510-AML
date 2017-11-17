import numpy as np
from scipy.optimize import minimize

class RidgeRegress(object):
    def __init__(self, reg_param):
        """
            Ridge Regress object constructor for Ridge Regression
            Args:
                reg_param       : float, regularization parameter (lambda in mathematical expressions)
        """        
        super(RidgeRegress, self).__init__()

        self.lmbda = reg_param
        print("RidgeRegress object: reg_param: {0}".format(self.lmbda))

    def fit(self, X, y, check_solution=False):
        """
            RidgeRegress fit method
            Args:
                X               : numpy.ndarray of shape (n_points, n_features) - Inputs
                y               : numpy.ndarray of shape (n_features) - Outputs/Targets
                check_solution  : (optional: Bool, default: False) check solution with sklearn's Ridge model
        """
        assert X.shape[0] == y.shape[0], "Mismatch input and output"
        self.theta = np.random.randn(X.shape[1])
        min_res = minimize(fun=objective, x0=self.theta, args=(X, y, self.lmbda), method='BFGS', options={'gtol': 1e-07})
        self.theta = min_res.x
        if check_solution:
            print("Distance between two solutions: {0}".format(compare_solution(self.theta, get_sklearn_sol(X, y, self.lmbda))))

    def predict(self, X):
        """
            RidgeRegress predict method
            Args:
                X               : numpy.ndarray of shape (n_points, n_features) - Inputs
            Returns:
                y               : Predictions in the same order
        """
        return np.dot(X, self.theta)
        
def objective(theta, *args):
    """
        Objective function for scipy.optimize.minimize
        Implements Ridge Regression
    """    
    X = args[0]
    y = args[1]
    lmbda = args[2]

    return np.power(np.dot(X, theta) - y, 2).sum() + lmbda*np.power(theta, 2).sum()

def compare_solution(X1, X2):
    """
        Simple util function
    """
    return np.linalg.norm(X1 - X2)

def get_sklearn_sol(X, y, reg_param):
    """
        Simple util function to get sklearn's Ridge model solution
    """
    from sklearn.linear_model import Ridge
    rr_sk = Ridge(alpha=reg_param, fit_intercept=False).fit(X, y)
    return rr_sk.coef_
