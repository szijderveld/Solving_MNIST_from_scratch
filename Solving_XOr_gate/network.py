


#Finaly we can make the newtowrk through defining the network class
class Network:
	def __init__(self):
		self.layers = []
		self.loss =None
		self.loss_prime = None 

	# add layer to network
	def add(self, layer):
		self.layers.append(layer)

	# define loss
	def use(self, loss, loss_prime):
		self.loss = loss
		self.loss_prime = loss_prime

	def predict(self, input_data):
		'''
		Work by inputting one row of data into output. Passing it through the layers and updates the value to the NN output.
		'''
		#sample dimension first (size of data)
		samples = len(input_data)
		result = []

		for i in range(samples):
			#forward propogation
			output = input_data[i]
			for layer in self.layers:
				output = layer.forward_propagation(output)
			result.append(output)

		return result

	#training
	def fit(self, x_train, y_train, epochs, learning_rate):
		#sample dimensions
		samples = len(x_train)


		#training loop
		for i in range(epochs):
			err = 0
			for j in range(samples):
				#forward propogation
				output = x_train[j]
				for layer in self.layers:
					output = layer.forward_propagation(output)

				#compute loss function (for tracking purposes only)
				err += self.loss(y_train[j], output)


				#backward propagation
				error = self.loss_prime(y_train[j], output)
				for layer in reversed(self.layers):
					error = layer.backward_propagation(error, learning_rate)


			#calculate average error on samples for that epoch
			err /= samples
			print('epoch %d/%d error=%f' % (i+1, epochs, err))  