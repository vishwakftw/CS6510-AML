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
		self.randomizer	= numpy.random.randint
		self.path_split	= os.path.split
		self.file_get	= numpy.genfromtxt
		self.joiner	= os.path.join
		
	def get_ref(self, field):
		if field == 'splits':
			return 0
		elif field == 'val':
			return 1
		elif field == 'indices':
			return 2
			
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
		return [splits, val, indices]
				
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
			self.features_type.append((type(X[0][i]).__name__ == 'str'))
			
			if type(X[0][i]).__name__ == 'str':
				self.features_set[i]	= list(set(self.np.array(X)[:,i].astype(type(X[0][i])).tolist()))
			
			elif type(X[0][i]).__name__ == 'int':

				if i == 0:
					self.features_bins[0]	= self.get_bins(0, 8)

				elif i == 2:
					self.features_bins[2]	= self.get_bins(2, 15)

				elif i == 4:
					self.features_bins[4]	= [[1, 3], [3, 5], [5, 8], [8, 11], [11, 13], [13, 17]]

				elif i == 10:
					self.features_bins[10]	= self.get_bins(10, 10)
				
				elif i == 11:
					self.features_bins[11]	= self.get_bins(11, 12)

				elif i == 12:
					self.features_bins[12]	= self.get_bins(12, 10)
				
		self.root		= self.node()
		self.root[self.get_ref('indices')]	= self.np.arange(0, len(X)).tolist()
		self.construct(self.root, depth, self.np.arange(0, 14).tolist())

		# Removing unwanted stuff
		del self.np
		del self.os
		del self.X
		del self.y
		
	def get_bins(self, feature_no, n_bins):
		"""
			Function to build bins for numeric data
			Args:
				feature_no	= Feature number ( 0 .... 13 )
				n_bins		= Division / Number of bins you want to divide into
		"""
		assert self.features_type[feature_no] == False, "Non-ints cannot be put in bins"
		
		temp_x	= self.np.array(self.X)
		max_val	= self.np.amax(temp_x[:, feature_no].astype(int))
		min_val	= self.np.amin(temp_x[:, feature_no].astype(int))

		# Remove this temporary data
		del temp_x
		
		bin_split	= (max_val - min_val)/n_bins + 1
		cur_low		= min_val - 1
		bins		= []
		while cur_low < max_val:
			bins.append([cur_low, cur_low + bin_split])
			cur_low	= cur_low + bin_split
			
		return bins
			
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
		
	def make_split(self, feature_no, X_indices):
		"""
			Function to make a split
			Args:
				feature_no	= the index of the feature
				X_indices	= the list of indices
			Returns:
				3D-list of shape
		"""
		if self.features_type[feature_no] == True:
			splits = []
			for _ in xrange(0, len(self.features_set[feature_no])):
				splits.append([[], []])

			for i in X_indices:
				for j in xrange(0, len(self.features_set[feature_no])):
					if self.features_set[feature_no][j] == self.X[i][feature_no]:
						splits[j][self.y[i]].append(i)
						
		elif self.features_type[feature_no] == False:
			splits	= []
			for i in xrange(0, len(self.features_bins[feature_no])):
				splits.append([[], []])
				
			for i in X_indices:
				for j in xrange(len(self.features_bins[feature_no])-1, -1, -1):
					if self.X[i][feature_no] >= self.features_bins[feature_no][j][0] and self.X[i][feature_no] < self.features_bins[feature_no][j][1]:
						splits[j][self.y[i]].append(i)
		return splits
		
	def get_best_split(self, X_indices, given_features):
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
		given_features		= self.np.random.permutation(given_features).tolist()
		for f_no in xrange(0, len(given_features)):
			# Split generated based on integer values taken
			# Typically the continuous fields fall in this category
			# Continuous values are bins
						
			# Split generated based on string type attributes
			# Typically categorical data falls in this category
			cur_split				= self.make_split(given_features[f_no], X_indices)
			cur_imp_split, cur_each_imp_split	= self.impurity_after_split(cur_split)
				
			if best_imp_after_split > cur_imp_split:
				best_split		= cur_split
				best_imp_after_split	= cur_imp_split
				best_each_imp_split	= cur_each_imp_split
				best_split_feature_no	= given_features[f_no]
		
		return best_split, best_each_imp_split, best_split_feature_no
		
	def construct(self, some_node, depth, available_features):
		"""
			Function to construct a subtree from a given node (inplace)
			Args:
				some_node		= node of the tree
				depth			= depth of the tree
				available_features	= available features to check for
		"""
		if depth > 0:
			if len(available_features) == 0:
				count_0 = count_1 = 0
				for i in xrange(0, len(some_node[self.get_ref('indices')])):
					if self.y[i] == 0:
						count_0	+= 1
					elif self.y[i] == 1:
						count_1 += 1
				some_node[self.get_ref('val')] = 1 if count_1 > count_0 else 0
				if len(some_node[self.get_ref('indices')]) == 0:
					some_node[self.get_ref('val')] = 1 if self.np.random.randint(1, 1000) % 5 == 3 else 0

				del some_node[self.get_ref('indices')]
			else:	
				new_split_attr		= self.get_best_split(some_node[self.get_ref('indices')], available_features)
				f_no			= new_split_attr[2]
				split_achieved		= new_split_attr[0]
				imp_at_each_split	= new_split_attr[1]
				del new_split_attr
				
				some_node[self.get_ref('val')] 	= (f_no, self.features_type[f_no])
				some_node[self.get_ref('splits')]	= []

				del some_node[self.get_ref('indices')]

				for i in xrange(0, len(split_achieved)):
					linear_indices	= split_achieved[i][0] + split_achieved[i][1]
					some_node[self.get_ref('splits')].append(self.node(indices=linear_indices))
	
				for i in xrange(0, len(split_achieved)):
					if round(imp_at_each_split[i], 5) == 0.0:
						some_node[self.get_ref('splits')][i][self.get_ref('val')]	= 0 if len(split_achieved[i][0]) != 0 else 1
					else:
						self.construct(some_node[self.get_ref('splits')][i], depth-1, list(filter(lambda x: x != f_no, available_features)))
						
		else:
			count_0	= count_1 = 0
			for i in xrange(0, len(some_node[self.get_ref('indices')])):
				if self.y[i] == 0:
					count_0 += 1
				elif self.y[i] == 1:
					count_1 += 1
			some_node[self.get_ref('val')] = 1 if count_1 > count_0 else 0
			if len(some_node[self.get_ref('indices')]) == 0:
				some_node[self.get_ref('val')] = 1 if self.np.random.randint(1, 1000) % 5 == 3 else 0

			del some_node[self.get_ref('indices')]
				
	def get_dataset(self, root_folder, filename, train=True):
		"""
			Function to get the dataset from the root folder destination
			Args:
				root_folder		= Root Folder path. Can be relative
				filename		= Filename of the file
			Returns:
				X, Y 		 	= Inputs, and Outputs
		"""
		full_path	= self.joiner(root_folder, filename)
		raw_data	= self.file_get(full_path, delimiter=',', dtype=str)
		
		full_x	= []	
		if train == True:
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
			if train == True:
				full_y.append(int(raw_data[i][14]))
			
		if train == True:
			return full_x, full_y
		else:
			return full_x
		
	def predict(self, file_name):
		"""
			Function to predict the inputs in filename
			Args:
				file_name	= string of path to file
			Returns:
				predictions	= predictions made for the input in the file
		"""
		path_to_file	= self.path_split(file_name)
		x		= self.get_dataset(path_to_file[0], path_to_file[1], train=False)
		predictions	= self.predict_impl(X=x)
		return predictions
		
	def predict_impl(self, X):
		"""
			Implicit Function to predict the outputs for a given X
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
		if isinstance(some_node[self.get_ref('val')], tuple) == False:
			return some_node[self.get_ref('val')]
		else:
			# String attributes
			if some_node[self.get_ref('val')][1] == True:
				index	= None
				feature_no	= some_node[self.get_ref('val')][0]

				for i in xrange(0, len(self.features_set[feature_no])):
					if self.features_set[feature_no][i] == some_data[feature_no] :	
						index = i
						break
				
			# Integer attributes
			elif some_node[self.get_ref('val')][1] == False:
				index	= None
				feature_no	= some_node[self.get_ref('val')][0]

				for i in xrange(0, len(self.features_bins[feature_no])):
					if some_data[feature_no] >= self.features_bins[feature_no][i][0] and some_data[feature_no] < self.features_bins[feature_no][i][1] :
						index = i
						break

			# If the given instance is an entirely new path not seen by the decision tree, then guess
			if index is None:
				return 1 if self.randomizer(1, 10000) % 5 == 3 else 0
			else:
				return self.traverse(some_data, some_node[self.get_ref('splits')][index])
