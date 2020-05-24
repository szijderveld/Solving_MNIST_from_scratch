from layer import Layer 
import numpy as np 



#Fuild Fully Connected layer (linear / no activation layer)
class FCLayer(Layer):
	#input_size = numpy of input neurons
	#output_size = number of output neurons

	def __init__(self, input_size, output_size):
		self.weights = np.random.rand(input_size, output_size) - 0.5       #ouput between 0 and 1
		self.bias = np.random.rand(1, output_size) -0.5

	def forward_propagation(self, input_data):
		self.input = input_data
		self.output = np.dot(self.input, self.weights) + self.bias
		return self.output
	def backward_propagation(self, output_error, learning_rate):
		input_error = np.dot(output_error, self.weights.T)
		
		weights_errror = np.dot(self.input.T, output_error)

		#dBias = output_error


		#update parameters
		self.weights -= learning_rate * weights_errror
		self.bias -= learning_rate * output_error
		return input_error


