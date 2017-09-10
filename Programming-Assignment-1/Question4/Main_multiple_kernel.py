import numpy as np
from time import clock
from sklearn import svm
from dataloader import get_dataset
from argparse import ArgumentParser
from multi_kernels import MultiKernelheuristic as mkh
from multi_kernels import MultiKernelfixedrules as mkfr
from sklearn.model_selection import StratifiedKFold

parser	= ArgumentParser()
parser.add_argument('--dataroot', required=True, help='root folder for the dataset')
parser.add_argument('--kernel_set', required=True, help='kernel_set : fixed_rules | heuristic')
parser.add_argument('--gamma0', type=float, help='parameter for linear kernel')
parser.add_argument('--gamma1', type=float, help='parameter for polynomial kernel')
parser.add_argument('--gamma2', type=float, help='parameter for gaussian kernel')
parser.add_argument('--normalize', action='store_true', help='toggle for normalizing the input data')
opt	= parser.parse_args()
	
train, test_x		= get_dataset(root=opt.dataroot, normalize=opt.normalize)
train_x, train_y	= train[ : , 0 : train.shape[1] - 1], train[ : , train.shape[1] - 1 ]

cross_valid	= StratifiedKFold(n_splits=5)

tr_accuracies	= []
va_accuracies	= []
tr_times	= []

for tr, va in cross_valid.split(train_x, train_y):
	
	print('Started cross validation split: {0}'.format(len(tr_accuracies) + 1))
	print('Ratio: {0}/{1} :: TR/VA'.format(tr.shape[0], va.shape[0]))

	tr_x, va_x	= train_x[tr], train_x[va]
	tr_y, va_y	= train_y[tr], train_y[va]

	svm_classifier	= svm.SVC(kernel='precomputed')

	if opt.kernel_set == 'fixed_rules':
		multi_k	= mkfr(gammas={'linear': opt.gamma0, 'polynomial': opt.gamma1, 'gaussian': opt.gamma2},
				hyperparameters={'polynomial': 3, 'gaussian': 0.2})

	elif opt.kernel_set == 'heuristic':
		multi_k	= mkh(hyperparameters={'polynomial': 3, 'gaussian': 0.2})

	t_start	= clock()
	tr_gram_matrix	= multi_k.gram_matrix(tr_x, tr_x)

	svm_classifier.fit(tr_gram_matrix, tr_y)
	t_stop		= clock()
	
	va_gram_matrix	= multi_k.gram_matrix(va_x, tr_x)
	
	tr_times.append((t_stop - t_start))

	tr_predictions	= svm_classifier.predict(tr_gram_matrix)
	va_predictions	= svm_classifier.predict(va_gram_matrix)

	tr_accuracy	= 1 - np.abs(tr_predictions - tr_y).sum()/tr_y.shape[0]
	va_accuracy	= 1 - np.abs(va_predictions - va_y).sum()/va_y.shape[0]
	
	tr_accuracies.append(float(round(tr_accuracy, 5)))
	va_accuracies.append(float(round(va_accuracy, 5)))
	
file_results	= open('results.txt', 'a')
file_results.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n'.format(opt.kernel_set, round(np.mean(tr_accuracy), 4), round(np.std(tr_accuracy), 4), round(np.mean(va_accuracy), 4), round(np.std(va_accuracy), 4), round(np.mean(tr_times), 4)))
