# Solving the XOr Problem

For context on the problem I would adivse look here (https://medium.com/@jayeshbahire/the-xor-problem-in-neural-networks-50006411840b). 

If following the solution I would advise to explore the files in this order.

1. layer.py
This file contains the parent class for the layer of the NN. It contains the empty variables for weights and biases aswell as structure for forward and backward propogation functions.

2. fc_layer.py
Defines the weights and biases variables for a FCLayer as well as defines the propogation.

3. losses.py, activation.py
Defines the loss function and activation functions that could be utilised in the NN. These files can be altered to change the functions used.

3. activation_layer.py
This creates an activation layer which is applied to the output on the hidden layer. This layer is neccessary for the model to 'learn'.

4. network.py
Contains the network class. This hold class funcitnos to build the NN layers, fit the network and predict. 


Finally, these files are utilised in XOr_solver.py to solve the simple XOr gate problem. The solving of this problem shows that the NN is capable of learning.
