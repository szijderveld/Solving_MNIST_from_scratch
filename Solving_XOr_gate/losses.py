import numpy as np
'''
For the last layer we cannot get the differentialfrom the previous layer. So we need to define it manually using a loss functions such as mean Squared error (MSE)
'''
def mse(y_true, y_pred):
	return np.mean(np.power(y_true-y_pred,2 ));

def mse_prime(y_true, y_pred):
	return 2*(y_pred-y_true)/y_true.size;