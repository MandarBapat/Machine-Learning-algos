#!/usr/bin/env python
# coding: utf-8

# In[3]:


from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
cols=["Exam1","Exam2","Selected"]
data=pd.read_csv("marks1.csv",names=cols)
X_train=np.array(data.loc[:49,"Exam1":"Exam2"]).reshape(2,50)
Y_train=np.array(data.loc[:49,"Selected"]).reshape(1,50)
X_test=np.array(data.loc[50:,"Exam1":"Exam2"]).reshape(2,50)
Y_test=np.array(data.loc[50:,"Selected"]).reshape(1,50)

print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)

def sigmoid(Z):
    A=1/(1+np.exp(-Z))
    return A
def initialize_parameters(size):
    W=np.zeros((size,1))
    b=0
    parameters={"W":W,"b":b}
    return parameters

def forward_prop(X_train,parameters):
    Z=np.dot(parameters["W"].T,X_train) + parameters["b"]
    A=sigmoid(Z)
    return Z,A

def derivatives(Z,A,Y_train,X_train,parameters,m):
    dW=(1/m)*np.dot(X_train,(A-Y_train).T)
    db=(1/m)*np.sum(A-Y_train)
    grads={"dW":dW,"db":db}
    return grads

def compute_cost(A,Y_train,m):
    p= -(Y_train*np.log(A+1e-20) + (1-Y_train)*(np.log(1-A+1e-20)))
    J=(1/m)*(np.sum(p))
    return J

def perform_Logistic(X_train,Y_train,n_iterations,rate):
    parameters=initialize_parameters(X_train.shape[0])
    list1=[]
    for i in range(1,n_iterations+1):
        Z,A=forward_prop(X_train,parameters)
        J=compute_cost(A,Y_train,X_train.shape[1])
        list1.append(J)
        if(i%100==0):
            print("The cost after " + str(i) + "th "+"iteration is : ",J)
        grads=derivatives(Z,A,Y_train,X_train,parameters,X_train.shape[1])
        parameters["W"]=parameters["W"]-rate*grads["dW"]
        parameters["b"]=parameters["b"]-rate*grads["db"]
        
    return parameters,list1

def predict(X_test,parameters):
    Z=np.dot(parameters["W"].T,X_test)+parameters["b"]
    A=sigmoid(Z)
    A=np.int64(A>0.5)
    return A

n_iterations=5000
parameters,list1=perform_Logistic(X_train,Y_train,n_iterations,0.000001)
costs=np.array(list1)
iterations=np.arange(1,n_iterations+1)
Y_predicted=predict(X_test,parameters)
print(np.mean(Y_predicted==Y_test)*100)
plt.plot(iterations,costs,"b")
plt.xlabel("Number of iterations")
plt.ylabel("Cost")
plt.title("Plot of Cost function")
plt.show()
plt.scatter(iterations,costs,color="b")
plt.xlabel("Number of iterations")
plt.ylabel("Cost")
plt.title("Plot of Cost function")
plt.show()


# In[ ]:




