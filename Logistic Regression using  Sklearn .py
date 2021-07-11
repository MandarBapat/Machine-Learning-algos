#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

data=pd.read_csv("marks1.csv",header=None)
X = data.iloc[:, :-1]

    # y = target values, last column of the data frame
y = data.iloc[:, -1]

    # filter out the applicants that got admitted
admitted = data.loc[y == 1]

    # filter out the applicants that din't get admission
not_admitted = data.loc[y == 0]

X = np.c_[np.ones((X.shape[0], 1)), X]
y = y[:, np.newaxis]

model = LogisticRegression()
model.fit(X, y)
predicted_classes = model.predict(X)
accuracy = accuracy_score(y.flatten(),predicted_classes)
parameters = model.coef_
print(accuracy)

