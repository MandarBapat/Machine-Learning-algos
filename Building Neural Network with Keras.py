#!/usr/bin/env python
# coding: utf-8

# In[45]:


import numpy as np
from matplotlib import pyplot as plt 
import sklearn
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd


# In[10]:


#reading a csv file
df = pd.read_csv("housepricedata.csv")
print(df)


# In[11]:


#converting to numpy arrays
data = df.values.copy()
print(data)


# In[35]:


#splitting into input features and output
X=data[:,0:10].copy()
Y=data[:,-1].copy()
print(X)
print(X.shape)
print(Y)
print(Y.shape)


# In[36]:


#scaling input features
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
print(X_scale.shape)


# In[37]:


#train dev test split

X_train,X_dev_and_test,Y_train,Y_dev_and_test = train_test_split(X_scale,Y,test_size=0.3)

X_dev,X_test,Y_dev,Y_test = train_test_split(X_dev_and_test,Y_dev_and_test,test_size=0.5)

print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape,X_dev.shape,Y_dev.shape)


# In[38]:


#importing necessary keras packages
from keras.models import Sequential
from keras.layers import Dense


# In[40]:


#creating the model
model = Sequential([Dense(32,activation="relu",input_shape=(10,)),Dense(32,activation="relu"),Dense(1,activation="sigmoid")])


# In[41]:


#selecting the optimizer and loss function
model.compile(optimizer="sgd", loss="binary_crossentropy" , metrics=["accuracy"])


# In[43]:


# we train our model now
hist=model.fit(X_train,Y_train,batch_size=32,epochs=100,validation_data=(X_dev,Y_dev))


# In[44]:


# evaluating the model on test data
model.evaluate(X_test,Y_test)


# In[47]:


#visualizing the model1 through loss
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Model loss")
plt.legend(["Train loss","Val loss"],loc="upper right")
plt.show()


# In[49]:


#visualizing model1 through accuracy
plt.plot(hist.history["accuracy"])
plt.plot(hist.history["val_accuracy"])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Model accuracy")
plt.legend(["Train accuracy","Val accuracy"],loc="upper right")
plt.show()


# In[51]:


# creating model2
model2 = Sequential([Dense(1000,activation="relu",input_shape=(10,)),Dense(1000,activation="relu"),Dense(1000,activation="relu"),Dense(1000,activation="relu"),Dense(1,activation="sigmoid")])
model2.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
hist2=model2.fit(X_train,Y_train,batch_size=32,epochs=100,validation_data=(X_dev,Y_dev))


# In[55]:


#visualizing the model2 through loss
plt.plot(hist2.history["loss"])
plt.plot(hist2.history["val_loss"])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Model loss")
plt.legend(["Train loss","Val loss"],loc="upper right")
plt.show()


# In[52]:


#applying regularization
from keras.layers import Dropout
from keras import regularizers


# In[53]:


model3=Sequential([Dense(1000,activation="relu",input_shape=(10,),kernel_regularizer=regularizers.l2(0.01)),Dropout(0.3),Dense(1000,activation="relu",kernel_regularizer=regularizers.l2(0.01)),Dropout(0.3),Dense(1000,activation="relu",kernel_regularizer=regularizers.l2(0.01)),Dropout(0.3),Dense(1000,activation="relu",kernel_regularizer=regularizers.l2(0.01)),Dropout(0.3),Dense(1,activation="sigmoid",kernel_regularizer=regularizers.l2(0.01))])
model3.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
hist3=model3.fit(X_train,Y_train,batch_size=32,epochs=100,validation_data=(X_dev,Y_dev))


# In[54]:


#visualizing the model3 through loss
plt.plot(hist3.history["loss"])
plt.plot(hist3.history["val_loss"])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Model loss")
plt.legend(["Train loss","Val loss"],loc="upper right")
plt.show()


# In[ ]:




