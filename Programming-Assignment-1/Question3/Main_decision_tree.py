import pickle
import numpy as np
from random import shuffle
from decision_tree import DecisionTree

for depth in xrange(1, 8):
	for imp in ['gini', 'entropy']:
		for s in xrange(0, 25):
			model	= DecisionTree()
			x, y	= model.get_dataset('./data', 'train.csv')
			t	= [x[i] + [y[i]] for i in xrange(0, len(y))]
			shuffle(t)
			x	= [z[0:14] for z in t]
			y	= [z[14] for z in t]
			model.fit(X=x[:8000], y=y[:8000], depth=depth, impurity_func=imp)
			p_y	= model.predict_impl(X=x[8000:])
			accry	= 1 - np.abs(np.array(p_y) - np.array(y[8000:])).sum()/2000.0
			print 'Depth: {0}\tImpurity: {1}\tTrain Accuracy: {2}'.format(depth, imp, accry)
			del model
