#!/usr/bin/env python
# coding: utf-8

# In[9]:


from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


train_X, train_y = mnist.load_data()[0]
test_X, test_y = mnist.load_data()[1]
print(train_X.shape)
print(train_y.shape)


# In[3]:


(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(train_X.shape)
print(train_y.shape)


# In[7]:


m = 3
n = 3


# In[11]:


r = np.random.randint(0, train_X.shape[0], m*n)
print(r)


# In[15]:


for i in range(m*n):
    plt.subplot(m, n, i+1)
    plt.imshow(X_train[r[i]], cmap='gray', interpolation='none')
    plt.title("{}: Class {}".format(r[i], y_train[r[i]]))
    plt.axis('off')

plt.show()

# In[ ]:

