#from layer import Layer 
import numpy as np 

#Base Class for layer
class Layer:
	def __init__(self):
		self.input = None
		self.output = None

	# computes the ourput Y of a layer for a given input X
	def forward_propagation(self, input):
		raise NotImplementedError
	#computed dE/dX for a gien dE/dY (and update paratmers)
	def backward_propagation(self, output_error, learning_rate):
		raise NotImplementedError
