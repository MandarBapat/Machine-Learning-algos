#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os.path
from gensim import corpora
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt


# In[12]:


text = """Tesla gained experience in telephony and electrical engineering before emigrating to the United States in 1884 to work for Thomas Edison in New York City. He soon struck out on his own with financial backers, setting up laboratories and companies to develop a range of electrical devices. His patented AC induction motor and transformer were licensed by George Westinghouse, who also hired Tesla for a short time as a consultant. His work in the formative years of electric power development was involved in a corporate alternating current/direct current "War of Currents" as well as various patent battles.
"""


# In[13]:


stop_words = set(stopwords.words("english"))


# In[18]:


x = word_tokenize(text)
meaningful = []
for i in x:
    if i not in stop_words:
        meaningful.append(i)


# In[19]:


print(meaningful)


# In[ ]:




