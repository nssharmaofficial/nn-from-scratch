# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 12:10:48 2022

@author: Nataša Brisudová
"""

"""

L hidden layers (tanh,sigmoid)

"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn import preprocessing
import math


"""
Loading validation dataset
"""

#loading from excel and changing to numpy array
validation_x = pd.read_excel('validation_x.xlsx')
validation_x = validation_x.to_numpy()
validation_y = pd.read_excel('validation_y.xlsx')
validation_y = validation_y.to_numpy()

#normalizing dataset
for i in range(0,30):  # for each out of 31 variables
    validation_x[:,i] = preprocessing.normalize([validation_x[:,i]])

#transposing X set
validation_x = validation_x.T


"""
Loading train_plus_test dataset
"""

#loading from excel and changing to numpy array
train_plus_test_x = pd.read_excel('train_plus_test_x.xlsx')
train_plus_test_x = train_plus_test_x.to_numpy()
train_plus_test_y = pd.read_excel('train_plus_test_y.xlsx')
train_plus_test_y = train_plus_test_y.to_numpy()

#normalizing dataset
for i in range(0,30):
    train_plus_test_x[:,i] = preprocessing.normalize([train_plus_test_x[:,i]])
    
#transposing X set    
train_plus_test_x = train_plus_test_x.T


"""
Define functions
"""

def initialize_parameters_deep(layer_dims):
    
    """
    layer_dims -- python array containing the dimensions of each layer in the network
    
    Returns:
    parameters -- python dictionary containing parameters "W1", "b1", ..., "WL", "bL":
    """
    
    np.random.seed(14)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):

        parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        
        expit(parameters['W' + str(l)])
        expit(parameters['b' + str(l)])
        
        
    return parameters


# *************************ACTIVATION FUNCTIONS*************************

def sigmoid(Z):
    
    """
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, for backprop
    """
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

def sigmoid_backward(dA, cache):
    
    """
    Implement the backward propagation for a single SIGMOID unit.

    dA -- post-activation gradient
    cache -- 'Z' where we store for computing backprop 
    
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache 
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    assert (dZ.shape == Z.shape)
    return dZ


def tanh(Z):
    A = np.tanh(Z)
    cache = Z
    return A, cache

def tanh_backward(dA, cache):
 
    Z = cache
    A = np.tanh(Z)
    dZ = dA * (1 -np.power(A, 2))
    assert (dZ.shape == Z.shape)
    return dZ

# *************************FORWARD PROPAGATION*************************

def linear_forward(A, W, b):
    
    """
    A -- activations from previous layer (size of previous layer, number of examples)
    W -- weights matrix (size of current layer, size of previous layer)
    b -- bias vector (size of the current layer, 1)
    
    Returns:
    Z -- the input of the activation function
    cache -- a python dictionary containing "A", "W" and "b" for backprop
    """
    
    Z = W.dot(A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    A_prev -- activations from previous layer (size of previous layer, number of examples)
    W -- weights matrix (size of current layer, size of previous layer)
    b -- bias vector  (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a string: "sigmoid" or "tanh"
   
    Returns:
    A -- the output of the activation function
    cache -- a dictionary containing "linear_cache" and "activation_cache" for backprop
    """
    
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "tanh":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = tanh(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    
    """
    Implement forward propagation for the [LINEAR->TANH]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches 
    """

    caches = []
    A = X #first input
    L = len(parameters)//2  # number of layers in the neural network
    
    # Implement [LINEAR -> TANH]*(L-1)
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "tanh")
        caches.append(cache)
    
    # Implement LINEAR -> SIGMOID
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches

# *************************COST FUNCTION*************************

def compute_cost(AL, Y):
    
    """
    AL -- probability vector corresponding to label predictions (1, number of examples)
    Y -- true "label" vector (1, number of examples)
    
    Returns:
    cross-entropy cost
    """
    
    m = Y.shape[1] #number of examples

    # Compute loss from AL and y.
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    
    cost = np.squeeze(cost)      #  this turns [[17]] into 17
    assert(cost.shape == ())
    
    return cost

# *************************BACKWARD PROPAGATION*************************

def linear_backward(dZ, cache):
    
    """

    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- values (A_prev, W, b) coming from the forward prop in the current layer
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1)
    dW -- Gradient of the cost with respect to W (current layer l)
    db -- Gradient of the cost with respect to b (current layer l)
    """
    
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    
    # same shapes
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- values (linear_cache, activation_cache) we store for backprop
    activation -- the activation to be used in this layer, stored as string: "sigmoid" or "tanh"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1)
    dW -- Gradient of the cost with respect to W (current layer l)
    db -- Gradient of the cost with respect to b (current layer l)
    """
    linear_cache, activation_cache = cache
    
    if activation == "tanh":
        dZ = tanh_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    
    """
    Implement the backward propagation for the [LINEAR->TANH] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector
    caches 
    
    Returns:
    grads -- A dictionary with the gradients
    """
    
    grads = {}
    L = len(caches) # the number of layers
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Lth layer (SIGMOID -> LINEAR) gradients
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
    
    for l in reversed(range(L-1)):
        # lth layer: (TANH -> LINEAR) gradients
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "tanh")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

# *************************UPDATE PARAMETERS*************************

def update_parameters(parameters, grads, learning_rate):
    
    """
    parameters
    grads
    
    Returns:
    parameters -- dictionary containing updated parameters 

    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters


def random_mini_batches(X, Y, mini_batch_size, seed = 0):
    
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data (input size, number of examples)
    Y -- true "label" vector (1, number of examples)
    mini_batch_size -- size of the mini-batches
    
    Returns:
    mini_batches -- (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)            
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X[:,k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k * mini_batch_size:(k + 1) * mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:

        mini_batch_X = shuffled_X[:,num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:,num_complete_minibatches * mini_batch_size:]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

# *************************PREDICTION*************************

def predict(parameters, X):
    
    """
    parameters 
    X -- input data
    
    Returns
    predictions -- vector of predictions of our model
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.

    AL, caches = L_model_forward(X, parameters)
    predictions = np.round(AL)
    return predictions



# *************************MODEL*************************

def L_layer_model_minibatch (X, Y, layers_dims, mini_batch_size, learning_rate, num_iterations, print_cost=True):
    
    """
    Arguments:
    X -- dataset of shape (31, number of examples)
    Y -- labels of shape (1, number of examples)
    minibatch size --
            = m ==> Batch gradient descent 
            = 1 ==> Stochastic gradient descent (SGD)
            = between 1 and m ==> Mini-batch gradient descent
    layers_dims -- list containing the input size and each layer size
    learning_rate 
    num_iterations
    print_cost -- if True, print the cost every 100 iterations
    
    Returns:
    parameters --  used later to predict
    """
    
    np.random.seed(2)
    costs = []      
    seed = 10                 
    
    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)
    if mini_batch_size > 1:
        minibatch_X = X[:, mini_batch_size-1]
        minibatch_Y = Y[:, mini_batch_size-1]
    
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):
    
        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
         
        for minibatch in minibatches:
            
            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch
         
            # Forward propagation: [LINEAR -> TANH]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = L_model_forward(minibatch_X, parameters)
        
            # Compute cost.
            cost = compute_cost(AL, minibatch_Y)
    
            # Backward propagation.
            grads = L_model_backward(AL, minibatch_Y, caches)
 
            # Update parameters.
            parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))

        if print_cost and i % 100 == 0:
            costs.append(cost)
       
    # plot the cost

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate)
              +"\nMinibatch size = " +str((mini_batch_size))
              +"\nLayers = " +str(layers_dims))
    plt.show()

    return parameters


#*******************************************************************

parameters = L_layer_model_minibatch(train_plus_test_x, train_plus_test_y, layers_dims = [31,6,1],mini_batch_size=80, learning_rate = 0.005, num_iterations=20000, print_cost=True)


# Predict test/train set examples 
Y_prediction_validation = predict(parameters, validation_x)
Y_prediction_trainandtest = predict(parameters, train_plus_test_x)

# Print train/test Errors
print("validation accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_validation - validation_y)) * 100))
print("train plus test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_trainandtest - train_plus_test_y)) * 100))















