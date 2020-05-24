'''
Lets use this NN to solve XOr logic gate. This is a good test to see if a data set is learning

'''

import numpy as np 

from layer import Layer 
from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime
from losses import mse, mse_prime


#training data
x_train = np.array([[[0,0]],[[0,1]],[[1,0]],[[1,1]]])
y_train = np.array([[[0]],[[1]],[[1]],[[0]]])

print(x_train[0])

exit()
#network
net = Network()
net.add(FCLayer(2,3))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(3,1))
net.add(ActivationLayer(tanh, tanh_prime))


#train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate = 0.1)


#test
out = net.predict(x_train)
print(out)



#equivilant using sklearn
from sklearn.neural_network import MLPClassifier
x_train = [[0,0],[0,1],[1,0],[1,1]]
y_train = [[0],[1],[1],[0]]



clf = MLPClassifier( activation= 'tanh', solver = 'sgd', learning_rate_init=0.1)
clf.fit(x_train,y_train)
print(clf.predict(x_train))
print(clf.n_layers_)



