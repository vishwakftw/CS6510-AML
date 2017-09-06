import numpy as np
from sklearn import svm
from dataloader import get_dataset
from argparse import ArgumentParser
from kernels import linear_gram_matrix as lgm
from kernels import gaussian_gram_matrix as ggm
from kernels import polynomial_gram_matrix as pgm
from sklearn.model_selection import StratifiedKFold

parser	= ArgumentParser()
parser.add_argument('--dataroot', required=True, help='root folder for the dataset')
parser.add_argument('--kernel', required=True, help='kernel : linear | polynomial | gaussian')
parser.add_argument('--q', type=int, default=2, help='parameter for polynomial kernel')
parser.add_argument('--sigma', type=float, default=1, help='parameter for gaussian kernel')
opt	= parser.parse_args()
	
train, test_x		= get_dataset(root=opt.dataroot)
train_x, train_y	= train[ : , 0 : train.shape[1] - 1], train[ : , train.shape[1] - 1 ]

svm_classifier	= svm.SVC(kernel='precomputed')
cross_valid	= StratifiedKFold(n_splits=5)

tr_accuracies	= []
va_accuracies	= []

for tr, va in cross_valid.split(train_x, train_y):
	
	print('Started cross validation split: {0}'.format(len(tr_accuracies) + 1))	
	tr_x, va_x	= train_x[tr], train_x[va]
	tr_y, va_y	= train_y[tr], train_y[va]

	if opt.kernel == 'linear':
		tr_gram_matrix	= lgm(tr_x, tr_x)
		va_gram_matrix	= lgm(va_x, tr_x)
	elif opt.kernel == 'polynomial':
		tr_gram_matrix	= pgm(tr_x, tr_x, opt.q)
		va_gram_matrix	= pgm(va_x, tr_x, opt.q)
	elif opt.kernel	== 'gaussian':
		tr_gram_matrix	= ggm(tr_x, tr_x, opt.sigma)
		va_gram_matrix	= ggm(va_x, tr_x, opt.sigma)

	svm_classifier.fit(tr_gram_matrix, tr_y)
	tr_predictions	= svm_classifier.predict(tr_gram_matrix)
	va_predictions	= svm_classifier.predict(va_gram_matrix)

	tr_accuracy	= 1 - np.abs(tr_predictions - tr_y).sum()/tr_y.shape[0]
	va_accuracy	= 1 - np.abs(va_predictions - va_y).sum()/va_y.shape[0]
	
	tr_accuracies.append(round(tr_accuracy, 5))
	va_accuracies.append(round(va_accuracy, 5))
	
print('Train Accuracies across all folds: {0}'.format(tr_accuracies))
print('Validation Accuracies across all folds: {0}'.format(va_accuracies))
