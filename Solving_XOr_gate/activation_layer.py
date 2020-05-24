
from layer import Layer
# For our model to learn anything we need to aply non-linear functions to the output of our functions
class ActivationLayer(Layer):
	def __init__(self, activation, activation_prime):
		self.activation = activation
		self.activation_prime = activation_prime

	#returns the activation input
	def forward_propagation(self, input_data):
		self.input = input_data
		self.output = self.activation(self.input)
		return self.output

	#return input_error = dE/dX for a given for a given output_error = dE/dY.
	#learning_rate is not used because there is no 'learnable' parameters.
	def backward_propagation(self, output_error, learning_rate):
		return self.activation_prime(self.input) * output_error




