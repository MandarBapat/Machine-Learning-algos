#!/usr/bin/env python
# coding: utf-8

# # **LET'S DIVE INTO SOME IMPLEMENTATION**
# 
# 
# 
# 

# In[1]:


import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn


# ##Task 1 - Gradient Descent
#  
# 
# **Warm-up exercise**: Implement the gradient descent update rule. The  gradient descent rule is, for $l = 1, ..., L$: 
# $$ W^{[l]} = W^{[l]} - \alpha \text{ } dW^{[l]} \tag{1}$$
# $$ b^{[l]} = b^{[l]} - \alpha \text{ } db^{[l]} \tag{2}$$
# 
# where L is the number of layers and $\alpha$ is the learning rate. All parameters should be stored in the `parameters` dictionary. Note that the iterator `l` starts at 0 in the `for` loop while the first parameters are $W^{[1]}$ and $b^{[1]}$. You need to shift `l` to `l+1` when coding.

# In[4]:


# GRADED FUNCTION: update_parameters_with_gd

def update_parameters_with_gd(parameters, grads, learning_rate):
    """
    Update parameters using one step of gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters to be updated:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients to update each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    learning_rate -- the learning rate, scalar.
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """

    L = len(parameters) // 2 # number of layers in the neural networks

    # Update rule for each parameter
    for l in range(L):
        ### START CODE HERE ### (approx. 2 lines)

        parameters["W" + str(l+1)] = parameters["W"+str(l+1)]-learning_rate*grads["dW"+str(l+1)]
        parameters["b" + str(l+1)] = parameters["b"+str(l+1)]-learning_rate*grads["db"+str(l+1)]
        ### END CODE HERE ###
        
    return parameters


# In[5]:



#assign the value of parameters and grads.Do it in this cell itself so that the resulting parameters don't change every time you run this cell
parameters={'W1': array([[ 1.39134536, -0.61175641, -0.52817175],
         [-1.07296862,  0.86540763, -2.7815387 ]]),
  'W2': array([[ 0.3190391 , -0.24937038,  1.46210794],
         [-2.06014071, -0.3224172 , -0.38405435],
         [ 1.13376944, -1.09989127, -0.17242821]]),
  'b1': array([[ 1.74481176],
         [-0.7612069 ]]),
  'b2': array([[-0.87785842],
         [ 0.04221375],
         [ 0.58281521]])}
grads={'dW1': array([[-1.10061918,  1.14472371,  0.90159072],
         [ 0.50249434,  0.90085595, -0.68372786]]),
  'dW2': array([[-0.21788808,  0.53035547, -0.69166075],
         [-0.39675353, -0.6871727 , -0.84520564],
         [-0.67124613, -0.0126646 , -1.11731035]]),
  'db1': array([[-0.12289023],
         [-0.93576943]]),
  'db2': array([[ 0.2844157 ],
         [ 1.78980218],
         [ 0.79204416]])}


parameters = update_parameters_with_gd(parameters, grads, 0.001)
print("W1 =\n" + str(parameters["W1"]))
print("b1 =\n" + str(parameters["b1"]))


# # SECOND TASK

# ## 3 - Momentum
# 
# 
# 
# 
# 
# **Exercise**: Initialize the velocity. The velocity, $v$, is a python dictionary that needs to be initialized with arrays of zeros. Its keys are the same as those in the `grads` dictionary, that is:
# for $l =1,...,L$:
# ```python
# v["dW" + str(l+1)] = ... #(numpy array of zeros with the same shape as parameters["W" + str(l+1)])
# v["db" + str(l+1)] = ... #(numpy array of zeros with the same shape as parameters["b" + str(l+1)])
# ```
# **Note** that the iterator l starts at 0 in the for loop while the first parameters are v["dW1"] and v["db1"] (that's a "one" on the superscript). This is why we are shifting l to l+1 in the `for` loop.

# In[23]:


def initialize_velocity(parameters):
    """
    Initializes the velocity as a python dictionary with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    
    Returns:
    v -- python dictionary containing the current velocity.
                    v['dW' + str(l)] = velocity of dWl
                    v['db' + str(l)] = velocity of dbl
    """
    
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    
    # Initialize velocity
    for l in range(L):
        ### START CODE HERE ### (approx. 2 lines)
        v["dW" + str(l+1)] = np.zeros(parameters["W"+str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b"+str(l+1)].shape) 
        ### END CODE HERE ###
        
    return v


# In[41]:



v = initialize_velocity(parameters)
print("v[\"dW1\"] =\n" + str(v["dW1"]))
print("v[\"db1\"] =\n" + str(v["db1"]))
print("v[\"dW2\"] =\n" + str(v["dW2"]))
print("v[\"db2\"] =\n" + str(v["db2"]))


# **Expected Output**:
# 
# ```
# v["dW1"] =
# [[ 0.  0.  0.]
#  [ 0.  0.  0.]]
# v["db1"] =
# [[ 0.]
#  [ 0.]]
# v["dW2"] =
# [[ 0.  0.  0.]
#  [ 0.  0.  0.]
#  [ 0.  0.  0.]]
# v["db2"] =
# [[ 0.]
#  [ 0.]
#  [ 0.]]
# ```

# **Exercise**:  Now, implement the parameters update with momentum. The momentum update rule is, for $l = 1, ..., L$: 
# 
# $$ \begin{cases}
# v_{dW^{[l]}} = \beta v_{dW^{[l]}} + (1 - \beta) dW^{[l]} \\
# W^{[l]} = W^{[l]} - \alpha v_{dW^{[l]}}
# \end{cases}\tag{3}$$
# 
# $$\begin{cases}
# v_{db^{[l]}} = \beta v_{db^{[l]}} + (1 - \beta) db^{[l]} \\
# b^{[l]} = b^{[l]} - \alpha v_{db^{[l]}} 
# \end{cases}\tag{4}$$
# 
# where L is the number of layers, $\beta$ is the momentum and $\alpha$ is the learning rate. All parameters should be stored in the `parameters` dictionary.  Note that the iterator `l` starts at 0 in the `for` loop while the first parameters are $W^{[1]}$ and $b^{[1]}$ (that's a "one" on the superscript). So you will need to shift `l` to `l+1` when coding.

# In[42]:


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    Update parameters using Momentum
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- python dictionary containing the current velocity:
                    v['dW' + str(l)] = ...
                    v['db' + str(l)] = ...
    beta -- the momentum hyperparameter, scalar
    learning_rate -- the learning rate, scalar
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- python dictionary containing your updated velocities
    """

    L = len(parameters) // 2 # number of layers in the neural networks
    
    # Momentum update for each parameter
    for l in range(L):
        
        ### START CODE HERE ### (approx. 4 lines)
        # compute velocities
        v["dW" + str(l+1)] = beta*v["dW"+str(l+1)]  + (1-beta)*grads["dW"+str(l+1)]
        v["db" + str(l+1)] = beta*v["db"+str(l+1)]  + (1-beta)*grads["db"+str(l+1)]
        # update parameters
        parameters["W" + str(l+1)] = parameters["W"+str(l+1)] - learning_rate*v["dW"+str(l+1)]
        parameters["b" + str(l+1)] = parameters["b"+str(l+1)] - learning_rate*v["db"+str(l+1)]
        ### END CODE HERE ###
        
    return parameters, v


# The parameters and grads are the same as in task 1 and v is the same as earlier i.e., initialized to zero.
# 
# ---
# 
# 
# 
# 
# 
# 

# In[43]:


#assign the values of arguments
parameters={'W1': array([[ 1.39134536, -0.61175641, -0.52817175],
         [-1.07296862,  0.86540763, -2.7815387 ]]),
  'W2': array([[ 0.3190391 , -0.24937038,  1.46210794],
         [-2.06014071, -0.3224172 , -0.38405435],
         [ 1.13376944, -1.09989127, -0.17242821]]),
  'b1': array([[ 1.74481176],
         [-0.7612069 ]]),
  'b2': array([[-0.87785842],
         [ 0.04221375],
         [ 0.58281521]])}
grads={'dW1': array([[-1.10061918,  1.14472371,  0.90159072],
         [ 0.50249434,  0.90085595, -0.68372786]]),
  'dW2': array([[-0.21788808,  0.53035547, -0.69166075],
         [-0.39675353, -0.6871727 , -0.84520564],
         [-0.67124613, -0.0126646 , -1.11731035]]),
  'db1': array([[-0.12289023],
         [-0.93576943]]),
  'db2': array([[ 0.2844157 ],
         [ 1.78980218],
         [ 0.79204416]])}


#run the function

parameters,v = update_parameters_with_momentum(parameters, grads, v,0.9,0.001)
print("v[\"dW1\"] =\n" + str(v["dW1"]))
print("v[\"db1\"] =\n" + str(v["db1"]))
print("v[\"dW2\"] =\n" + str(v["dW2"]))
print("v[\"db2\"] =\n" + str(v["db2"]))


# END OF ASSIGNMENT
# 
# 
# 
# ---
# 
# 
# 
# 
