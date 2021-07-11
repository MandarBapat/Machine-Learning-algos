#!/usr/bin/env python
# coding: utf-8

# In[1]:


print(3+5)


# In[2]:


print(3**2)


# In[3]:


print("Hello")


# In[7]:


import numpy as np
a=np.random.rand(3,2)
print(a)


# In[27]:


import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
np.random.seed(0)
x=np.random.rand(100,1)
y=np.random.rand(100,1)+2+3*x
plt.scatter(x,y,color="b")
plt.xlabel("Input feature")
plt.ylabel("Output")
plt.title("Data points")
plt.show()
data=pd.read_table("http://bit.ly/chiporders")
print(data)


# In[76]:


def initialize_parameters(size):
    W=np.zeros((size,1))
    b=0
    parameters={"W":W,"b":b}
    return parameters
def forward_prop(X,parameters):
    Z=np.dot(parameters["W"].T,X)+parameters["b"]
    return Z
def derivatives(Z,Y,X):
    dW=(1/X.shape[1])*(np.dot((Z-Y),X.T))
    db=(1/X.shape[1])*(np.sum(Z-Y))
    grads={"dW":dW,"db":db}
    return grads

def calculate_cost(Z,Y):
    J=(1/2*(X.shape[1]))*(np.sum((Z-Y)**2))
    return J

def perform_Linear(X_train,Y_train,n_iterations,rate):
    parameters=initialize_parameters(X_train.shape[0])
    list1=[]
    for i in range(1,n_iterations+1):
        Z=forward_prop(X_train,parameters)
        J=calculate_cost(Z,Y_train)
        list1.append(J)
        if ((i%100)==0):
            print("The cost after "+str(i)+"th"+" iteration is : ",J)
        grads=derivatives(Z,Y_train,X_train)
        parameters["W"]=parameters["W"]-(rate*grads["dW"])
        parameters["b"]=parameters["b"]-(rate*grads["db"])
    return parameters,list1

def predict(X_test,parameters):
    Z=np.dot(parameters["W"].T,X)+parameters["b"]
    return Z

X=x.reshape(1,100)
Y=y.reshape(1,100)
n_iterations=1000
parameters,list1=perform_Linear(X,Y,n_iterations,0.1)
Y_predicted=predict(X,parameters)
costs=np.array(list1)
costs=costs.reshape((len(list1),1))
iterations=np.arange(1,n_iterations+1)
iterations=iterations.reshape(n_iterations,1)
plt.plot(iterations,costs,color="b",label="graph")
plt.xlabel("No. of iterations")
plt.ylabel("Cost")
plt.title("Plot of Cost function")
plt.show()
plt.scatter(iterations,costs,color="b",label="graph")
plt.xlabel("No. of iterations")
plt.ylabel("Cost")
plt.title("Plot of Cost function")
plt.show()
plt.scatter(X,Y,color="b")
plt.show()
plt.scatter(X,Y_predicted,color="b")
plt.show()


# In[ ]:




