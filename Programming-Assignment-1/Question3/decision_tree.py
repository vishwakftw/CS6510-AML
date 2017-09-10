class DecisionTree:
	"""
		Class to implement a decision tree
	"""
	def __init__(self):
		"""
			Constructor for the class
		"""
		import numpy
		self.np		= numpy
		self.root	= self.node()
		
	def node(self, split=None, val=None):
		"""
			Typical node for a tree
			Args:
				split		= List of splits (Generalize for n-way split)
				val		= Value of the Node. Defaults to None
			Returns:
				Node dictionary
		"""
		return {'split': split, 'val': val}
				
	def fit(self, X, y, impurity_func='entropy'):
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
		self.features_set	= []
		for i in xrange(0, self.features_count):
			self.features_set.append(list(set(self.np.array(X)[:,i].tolist())))
			
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

		imp	= 0.0
		for s in splits:
			inc_this_split	= float(len(s[0]) + len(s[1]))
			if inc_this_split == 0:
				continue
			class_probs	= [len(s[0])/inc_this_split, len(s[1])/inc_this_split]
			imp		= imp + inc_this_split*self.impurity(class_probs)/total_incoming
				
		return imp
		
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
					val	-= self.np.log(cp)*cp

		elif self.impurity_func == 'gini':
			for cp in class_probs:
				val	+= cp*(1 - cp)
			
		return val
		
	def make_split(self, feature_no, value=None):
		"""
			Function to make a split
			Args:
				feature_no	= the index of the feature
				value		= If the feature is a real value, then this will facilitate the 2-way split (< or otherwise)
			Returns:
				3D-list of shape [n_splits, n_points_in_each_class, 2] representing the indexes of points per class
		"""
		if value is None:
			splits = []
			for i in xrange(0, len(self.features_set[feature_no])):
				splits.append([[], []])

			for i in xrange(0, len(self.X)):
				for j in xrange(0, len(self.features_set[feature_no])):
					if self.features_set[feature_no][j] == self.X[i][feature_no]:
						splits[j][self.y[i]].append(i)
						
		else:
			splits	= [[[],[]], [[],[]]]
			for i in xrange(0, len(self.X)):
				if self.X[i][feature_no] < value:
					splits[0][self.y[i]].append(i)
				else:
					splits[1][self.y[i]].append(i)
					
		return splits
		
	def get_best_split(self):
		"""
			Function to get the best split at a given node based on the some splitting techniques
			Args:
				No args
			Returns:
				Best achieved impurity after split, Best split, and if the best split was achieved on a
				continuous attribute, then that splitting value
		"""
		best_imp_after_split	= 1.1
		best_split		= None
		best_split_val		= None
		best_split_feature_no	= -1
		for f_no in xrange(0, self.features_count):

			# Split generated based on integer values taken
			# Typically the continuous fields fall in this category
			if type(self.X[0][f_no]).__name__ == 'int':

				# Splitting based on the distinct values of the attributes
				for f_val in self.features_set[f_no]:					
					cur_split	= self.make_split(f_no, f_val)
					cur_imp_split	= self.impurity_after_split(cur_split)
					
					if best_imp_after_split > cur_imp_split:
						best_split		= cur_split
						best_imp_after_split	= cur_imp_split
						best_split_val		= f_val
						best_split_feature_no	= f_no
						
				# Splitting based on the pair-wise average of any two distinct values of the attributes
				for i in xrange(0, len(self.features_set[f_no])):
					for j in xrange(i, len(self.features_set[f_no])):
						f_val		= (self.features_set[f_no][i] + self.features_set[f_no][j])/2.0
						cur_split	= self.make_split(f_no, f_val)
						cur_imp_split	= self.impurity_after_split(cur_split)
						
						if best_imp_after_split > cur_imp_split:
							best_split		= cur_split
							best_imp_after_split	= cur_imp_split
							best_split_val		= f_val
							best_split_feature_no	= f_no
							
				# Splitting based on total average of all distinct values of the attributes
				f_val	= 0.0
				f_val	= self.np.mean(self.features_set[f_no])
				cur_split	= self.make_split(f_no, f_val)
				cur_imp_split	= self.impurity_after_split(cur_split)
				
				if best_imp_after_split > cur_imp_split:
					best_split		= cur_split
					best_imp_after_split	= cur_imp_split
					best_split_val		= f_val	
					best_split_feature_no	= f_no			

			# Split generated based on string type attributes
			# Typically categorical data falls in this category
			elif type(self.X[0][f_no]).__name__ == 'str':
				cur_split	= self.make_split(f_no)
				cur_imp_split	= self.impurity_after_split(cur_split)
				
				if best_imp_after_split > cur_imp_split:
					best_split		= cur_split
					best_imp_after_split	= cur_imp_split
					best_split_val		= None
					best_split_feature_no	= f_no
					
		return best_imp_after_split, best_split, best_split_val, best_split_feature_no
