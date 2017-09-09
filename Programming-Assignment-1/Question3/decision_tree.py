class DecisionTree:
	"""
		Class to implement a decision tree
	"""
	def __init__(self):
		"""
			Constructor for the class
		"""
		self.root	= self.node()
		
	def node(self, right=None, left=None, val=None):
		"""
			Typical node for a tree
			Args:
				right		= Right Node. Defaults to None
				left		= Left Node. Defaults to None
				val		= Value of the Node. Defaults to None
			Returns:
				Node dictionary
		"""
		return {'right': right, 'left': left, 'val': val}
	
	def insert(self, some_node, x):
		"""
			Function to insert a node into a tree
			Args:
				x		= Value to insert
				some_node	= Root of subtree to insert into
			Returns:
				Modified tree
			"""
		if some_node['val'] is None:
			some_node['val'] = x
		else:
			if x > some_node['val']:
				if some_node['right'] is None:
					some_node['right'] = self.node(val=x)
				else:
					some_node['right'] = self.insert(some_node['right'], x)
			else:
				if some_node['left'] is None:
					some_node['left'] = self.node(val=x)
				else:
					some_node['left'] = self.insert(some_node['left'], x)
		return some_node
		
	def display(self, some_node):
		"""
			Returns pre-order traversal display
		"""
		print some_node['val']
		if some_node['left'] is not None:
			self.display(some_node['left'])
		if some_node['right'] is not None:
			self.display(some_node['right'])
