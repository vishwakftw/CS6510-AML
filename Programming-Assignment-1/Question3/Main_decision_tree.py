import pickle
import numpy as np
from random import shuffle
from decision_tree import DecisionTree

results_gini	= []
results_entropy	= []

for imp in ['gini', 'entropy']:
	for depth in xrange(1, 8):
		cur_res	= []
		for s in xrange(0, 10):
			model	= DecisionTree()

			x, y	= model.get_dataset('./data', 'train.csv')
			t	= [x[i] + [y[i]] for i in xrange(0, len(y))]
			shuffle(t)

			x	= [z[0:14] for z in t]
			y	= [z[14] for z in t]

			model.fit(X=x[:8000], y=y[:8000], depth=depth, impurity_func=imp)

			p_y	= model.predict_impl(X=x[8000:])
			accry	= 1 - np.abs(np.array(p_y) - np.array(y[8000:])).sum()/2000.0
			cur_res.append(accry)

			del model

		if imp == 'gini':
			results_gini.append(cur_res)
		elif imp == 'entropy':
			results_entropy.append(cur_res)

results_gini_means	= []
for r in results_gini:
	results_gini_means.append(np.mean(r))

results_entropy_means	= []
for r in results_entropy:
	results_entropy_means.append(np.mean(r))
		
print results_gini_means
print results_entropy_means
