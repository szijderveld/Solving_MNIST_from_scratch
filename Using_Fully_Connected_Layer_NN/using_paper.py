
#pip install python-mnist

import gzip
import numpy as np
from catergorical import to_categorical
from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from losses import mse, mse_prime
from activations import tanh, tanh_prime



#training sample size 6000, test sample size 1000
image_size=28


def import_image(file_name, num_samples, image_size = 28):
	with gzip.open(file_name,'r') as f:
		f.read(16)
		buf = f.read(image_size * image_size * num_samples)
		data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
		#print(data.shape)
		#data = data.reshape(num_samples, image_size, image_size,1)
		data = data.reshape(num_samples,1, image_size*image_size)
	return data

def import_labels(file_name, num_samples):
	data = []
	with gzip.open(file_name,'r') as f:
		f.read(8)
		for i in range(0,num_samples):
			buf = f.read(1)
			labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
			data.append(labels)
	return data

import matplotlib.pyplot as plt
def visualise_image(data, num_image):
	image =  np.asarray(data[num_image].squeeze())
	plt.imshow(image)
	plt.show()


x_train = import_image('train-images-idx3-ubyte.gz', 6000)
x_train /= 255
y_train = import_labels('train-labels-idx1-ubyte.gz',6000)
x_test = import_image('t10k-images-idx3-ubyte.gz',1000)
x_test /= 255
y_test = import_labels('t10k-labels-idx1-ubyte.gz',1000)


y_train = to_categorical(y_train)							#copied from tensorflow.keras.np_utils
y_test = to_categorical(y_test)

#x_train = np.asarray(x_train)


#training 
net = Network()
net.add(FCLayer(image_size**2, 100))						#input (1, input_shape**2)  , output (1,100)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(100, 50))									#input (1,100 )  , output (1, 50)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(50,10))										#input (1,50) ,   output(1,10)
net.add(ActivationLayer(tanh, tanh_prime))						



net.use(mse, mse_prime)
net.fit(x_train[0:1000], y_train[0:1000], epochs =35, learning_rate=0.1)




