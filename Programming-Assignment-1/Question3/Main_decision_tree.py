from numpy.random import randint
from decision_tree import DecisionTree

model = DecisionTree()
for i in xrange(0, 10):
	model.insert(model.root, randint(0, 10000))
	
model.display(model.root)
