#!/usr/bin/env python
# coding: utf-8

# In[62]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score,mean_absolute_error
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor


# In[63]:


data=pd.read_table("winequality-red.csv",sep=";")
print(data.head())


# In[64]:


#solving Q1
data.isnull().sum()


# In[65]:


#solving Q2
p=data[(data["quality"]==5)|(data["quality"]==6)].copy()
percentage=round((x.shape[0]/data.shape[0])*100,2)
print(percentage)


# In[66]:


#solving Q3
r_values=data["pH"].max()-data["pH"].min()
print(round(r_values,2))


# In[67]:


#solving Q4
prop=(data[(data["quality"]>6)&(data["sulphates"]>0.65)].shape[0])/(data[data["quality"]>6].shape[0])
print(prop)


# In[68]:


#solving Q5
print(data[data["quality"]>6].describe())


# In[69]:


#solving Q6
print(32.572238**2)


# In[70]:


#splitting the data into test and train sets
X_train,X_test,Y_train,Y_test=train_test_split(data.drop("quality",axis=1),data["quality"],test_size=0.2,random_state=1)


# In[71]:


#solving Q7
linear=LinearRegression()
linear.fit(X_train,Y_train)
Y_predicted1=linear.predict(X_test)
mae=mean_absolute_error(Y_predicted1,Y_test)
mse=mean_squared_error(Y_predicted1,Y_test)
print(mse,mae)


# In[72]:


#solving Q8
support_regression=SVR(kernel="rbf",C=0.1,degree=4)
support_regression.fit(X_train,Y_train)
Y_predicted2=support_regression.predict(X_test)
print(mean_squared_error(Y_predicted2,Y_test))


# In[73]:


#solving Q9
logistic=LogisticRegression()
logistic.fit(X_train,Y_train)
Y_predicted3=logistic.predict(X_test)
print(Y_predicted3)


# In[74]:


#solving Q10
forest=RandomForestRegressor(n_estimators=200,max_depth=9,random_state=1,n_jobs=-1)
forest.fit(X_train,Y_train)
Y_predicted4=forest.predict(X_test)
print(mean_squared_error(Y_predicted4,Y_test))
print(mean_absolute_error(Y_predicted4,Y_test))

