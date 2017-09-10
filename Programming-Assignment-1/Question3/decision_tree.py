class DecisionTree:
	"""
		Class to implement a decision tree
	"""
	def __init__(self):
		"""
			Constructor for the class
		"""
		import os
		import numpy
		self.np		= numpy
		self.os		= os
		self.root	= self.node()
		
	def node(self, splits=None, val=None, indices=None):
		"""
			Typical node for a tree
			Args:
				split		= List of splits (Generalize for n-way split)
				val		= Value of the Node. Defaults to None
				indices		= List of indices of inputs coming to that node
			Returns:
				Node dictionary
		"""
		return {'splits': splits, 'val': val, 'indices': indices}
				
	def fit(self, X, y, depth=10, impurity_func='entropy'):
		"""
			Function called to generate a univariate decision tree out of a given dataset
			Args:
				X		= 2D-list of shape [n_points, n_features]
				Y		= 1D-list of shape [n_points]
				impurity_func	= Impurity function. Defaults to 'entropy'. Other option is 'gini'.
		"""
		assert impurity_func in ['entropy', 'gini'], "Invalid Impurity Functions."
		self.X			= X
		self.y			= y
		self.impurity_func	= impurity_func
		
		self.features_count	= len(X[0])
		self.features_set	= {}
		self.features_bins	= {}
		self.features_type	= []
		for i in xrange(0, self.features_count):
			self.features_type.append(type(X[0][i]).__name__)
			if type(X[0][i]).__name__ == 'str':
				self.features_set[i]	= list(set(self.np.array(X)[:,i].astype(self.features_type[i]).tolist()))
			elif type(X[0][i]).__name__ == 'int':
				self.features_bins[i]	= []
				if i == 0:
					bin_size	= 10
					low		= 16
					high		= 90
					while low < high:
						self.features_bins[i].append([low, low + high])
						low	= low + high

				elif i == 2:
					bin_size	= 77600
					low		= 19000
					high		= 1185000
					while low < high:
						self.features_bins[i].append([low, low + high])
						low	= low + high

				elif i == 4:
					self.features_bins[i]	= [[1, 3], [3, 5], [5, 8], [8, 11], [11, 13], [13, 17]]

				elif i == 10:
					bin_size	= 10000
					low		= 0
					high		= 100000
					while low < high:
						self.features_bins[i].append([low, low + high])
						low	= low + high
				
				elif i == 11:
					bin_size	= 400
					low		= 0
					high		= 4400
					while low < high:
						self.features_bins[i].append([low, low + high])
						low	= low + high
						
				elif i == 12:
					bin_size	= 10
					low		= 0
					high		= 100
					while low < high:
						self.features_bins[i].append([low, low + high])
						low	= low + high
				
		self.root		= self.node()
		self.root['indices']	= self.np.arange(0, len(X)).tolist()
		self.construct(self.root, depth)
			
	def impurity_after_split(self, splits):
		"""
			Function to calculate the impurity after a split.
			Args:
				splits		= 3D-list of shape [n_splits, n_points_in_each_class, 2]
			Returns:
				Scalar value: impurity after split
		"""
		total_incoming	= 0
		for s in splits:
			for cl in s:
				total_incoming	+= len(cl)

		tot_imp	= 0.0
		imp_each_split	= []
		for s in splits:
			inc_this_split	= float(len(s[0]) + len(s[1]))
			if round(inc_this_split, 5) == 0:
				imp_each_split.append(1.0)
			else:
				class_probs	= [len(s[0])/inc_this_split, len(s[1])/inc_this_split]
				node_imp	= self.impurity(class_probs)
				tot_imp		= tot_imp + inc_this_split*node_imp/total_incoming
				imp_each_split.append(node_imp)
				
		return tot_imp, imp_each_split
		
	def impurity(self, class_probs):
		"""
			Function to calculate the impurity of a node
			Args:
				class_probs	= Class Probability
			Returns:
				The impurity of a node
		"""
		assert sum(class_probs) == 1, "Class probabilities don't add up to 1"

		val	= 0.0	
		if self.impurity_func == 'entropy':
			for cp in class_probs:
				if round(cp, 5) == 0.0:
					val	-= 0
				else:
					val	-= self.np.log2(cp)*cp

		elif self.impurity_func == 'gini':
			for cp in class_probs:
				val	+= cp*(1 - cp)
			
		return val
		
	def make_split(self, feature_no, X_indices, value=None):
		"""
			Function to make a split
			Args:
				feature_no	= the index of the feature
				value		= If the feature is a real value, then this will facilitate the 2-way split (< or otherwise)
			Returns:
				3D-list of shape [n_splits, n_points_in_each_class, 2] representing the indexes of points per class
		"""
		if self.features_type[feature_no] == 'str':
			splits = []
			for i in xrange(0, len(self.features_set[feature_no])):
				splits.append([[], []])

			for i in X_indices:
				for j in xrange(0, len(self.features_set[feature_no])):
					if self.features_set[feature_no][j] == self.X[i][feature_no]:
						splits[j][self.y[i]].append(i)
						
		elif self.features_type[feature_no] == 'int':
			splits	= []
			for i in xrange(0, len(self.features_bins[feature_no])):
				splits.append([[], []])
				
			for i in X_indices:
				for j in xrange(len(self.features_bins[feature_no])-1, -1, -1):
					if self.X[i][feature_no] >= self.features_bins[feature_no][j][0] and self.X[i][feature_no] < self.features_bins[feature_no][j][1]:
						splits[j][self.y[i]].append(i)
						break
		return splits
		
	def get_best_split(self, X_indices):
		"""
			Function to get the best split at a given node based on the some splitting techniques
			Args:
				No args
			Returns:
				-> Best split
				-> Best impurity after split (Total impurity)
				-> Best impurity at each split node
				-> Best splitting val (None for categorial data, otherwise the value used for the binary split)
				-> Best feature to split
		"""
		best_imp_after_split	= 1.1
		best_each_imp_split	= None
		best_split		= None
		best_split_feature_no	= -1
		for f_no in xrange(0, self.features_count):
			print "At feature number : {0}".format(f_no)
			# Split generated based on integer values taken
			# Typically the continuous fields fall in this category
			# Continuous values are bins
						
			# Split generated based on string type attributes
			# Typically categorical data falls in this category
			cur_split				= self.make_split(f_no, X_indices)
			cur_imp_split, cur_each_imp_split	= self.impurity_after_split(cur_split)
				
			if best_imp_after_split > cur_imp_split:
				best_split		= cur_split
				best_imp_after_split	= cur_imp_split
				best_each_imp_split	= cur_each_imp_split
				best_split_feature_no	= f_no
					
		return best_split, best_imp_after_split, best_each_imp_split, best_split_feature_no
		
	def construct(self, some_node, depth):
		"""
			Function to construct a subtree from a given node
			Args:
				some_node	= node of the tree
		"""
		if depth > 0:
			new_split_attr		= self.get_best_split(some_node['indices'])
			some_node['val'] 	= (new_split_attr[3], self.features_type[new_split_attr[3]])
			some_node['splits']	= []

			for i in xrange(0, len(new_split_attr[0])):
				linear_indices	= new_split_attr[0][i][0] + new_split_attr[0][i][1]
				some_node['splits'].append(self.node(indices=linear_indices))
				if round(new_split_attr[2][i], 5) == 0.0 or len(linear_indices) == 1:
					if round(new_split_attr[2][i], 5) == 0.0:
						some_node['splits'][i]['val']	= 0 if len(new_split_attr[0][i][0]) != 0 else 1
					if len(linear_indices) == 1:
						some_node['splits'][i]['val']	= self.y[linear_indices[0]]
					continue
				else:
					print 'going to depth {0}'.format(depth-1)
					self.construct(some_node['splits'][i], depth-1)
		else:
			count_0	= count_1 = 0
			for i in xrange(0, len(some_node['indices'])):
				if self.y[i] == 0:
					count_0 += 1
				elif self.y[i] == 1:
					count_1 += 1
			some_node['val'] = 1 if count_1 > count_0 else 0
				
	def get_dataset(self, root_folder, filename):
		"""
			Function to get the dataset from the root folder destination
			Args:
				root_folder		= Root Folder path. Can be relative
				filename		= Filename of the file
			Returns:
				X, Y 		 	= Inputs, and Outputs
		"""
		full_path	= self.os.path.join(root_folder, filename)
		raw_data	= self.np.genfromtxt(full_path, delimiter=',', dtype=str)
		
		full_x	= []	
		full_y	= []

		for i in xrange(0, raw_data.shape[0]):

			# Replacements for unknown entries in raw_data with least occuring values
			if raw_data[i][1][1:] == '?':
				raw_data[i][1] 	= ' Never-worked' if i % 2 == 0 else ' Without-pay'
				
			# Same here
			if raw_data[i][6][1:] == '?':
				raw_data[i][6] 	= ' Armed-Forces' if i % 2 == 0 else ' Priv-house-serv'
				
			# Same here too
			if raw_data[i][13][1:] == '?':
				raw_data[i][13]	= ' Trinidad&Tobago' if i % 2 == 0 else ' Ecuador'

			full_x.append(
				[int(raw_data[i][0]), raw_data[i][1][1:], int(raw_data[i][2]), raw_data[i][3][1:], 
				int(raw_data[i][4]), raw_data[i][5][1:], raw_data[i][6][1:], raw_data[i][7][1:], 
				raw_data[i][8][1:], raw_data[i][9][1:], int(raw_data[i][10]), int(raw_data[i][11]), 
				int(raw_data[i][12]), raw_data[i][13][1:]]
			 	)
			full_y.append(int(raw_data[i][14]))
			
		return full_x, full_y
		
	def predict(self, X):
		"""
			Function to predict the outputs for a given X
			Args:
				X		= 2D-list of size (n_points, n_features)
			Returns:
				predictions	= predictions made for the set
		"""
		predictions	= []
		for i in xrange(0, len(X)):
			predictions.append(self.traverse(X[i], self.root))
			
		return predictions
		
	def traverse(self, some_data, some_node):
		"""
			Function to traverse the tree in order to get outputs
			Args:
				some_data	= 1D list representing a feature vector/input
				some_node	= the current node you are at
			Returns:
				0/1 based on tree structure
		"""
		if isinstance(some_node['val'], tuple) == False:
			return some_node['val']
		else:
			if some_node['val'][1] == 'str':
				index	= None
				for i in xrange(0, len(self.features_set[some_node['val'][0]])):
					if self.features_set[some_node['val'][0]][i] == some_data[some_node['val'][0]] :	
						index = i
						break
				
			else:
				index	= None
				for i in xrange(0, len(self.features_bins[some_node['val'][0]])):
					if some_data[some_node['val'][0]] >= self.features_bins[some_node['val'][0]][i][0] and some_data[some_node['val'][0]] < self.features_bins[some_node['val'][0]][i][1] :
						index = i
						break

			return self.traverse(some_data, some_node['splits'][i])			
