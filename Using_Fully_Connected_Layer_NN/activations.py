import numpy as np


#this activation layer uses a activation fuction and its derivative such as
def tanh(x):
	return np.tanh(x);
def tanh_prime(x):
	return 1-np.tanh(x)**2;