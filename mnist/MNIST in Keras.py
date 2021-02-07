#!/usr/bin/env python
# coding: utf-8

# In[1]:


# # Building a simple neural-network with Keras
# 
# **Author: Xavier Snelgrove**
# 
# This is a simple quick-start in performing digit recognition in a neural network in Keras, for a short tutorial at the University of Toronto. It is largely based on the `mnist_mlp.py` example from the Keras source.
# 

# ## Install prerequisites
# First steps (instructions for Mac or Linux). You need to install a recent version of Python, plus the packages `keras`, `numpy`, `matplotlib` and `jupyter`.
# 
# ### Install a recent Python
# 
# If you haven't installed a recent Python I recommend installing via Homebrew on a Mac from http://brew.sh and then installing Python via `brew install python`.
# 
# 
# ### Configure a virtual environment
# 
# You can install the packages globally, but I suggest installing them in a `virtualenv` virtual environment that basically encapsulates a full isolated Python environment. First you'll need to install a Python package manager called `pip` thus:
# 
#     easy_install pip
#     
# (If you get a permissions error, try adding a `sudo` to the beginning, so `sudo easy_install pip`)
# 
# Now install virtualenv thus:
# 
#     pip install virtualenv
# 
# Navigate to your home directory `cd ~` and create a virtual environment. We'll call it `kerasenv`
# 
#     virtualenv kerasenv
# 
# Now, to switch your shell environment to be within the env:
# 
#     source kerasenv/bin/activate
#     
# Great: now you can install the other prerequisites into this environment.
# 
#     pip install numpy jupyter keras matplotlib
#     
# 
# ## Open a new notebook
# 
# Now that everything's installed, you can open one of these web-based Python environments with the following command:
# 
#     ipython notebook
#     
# Create a new Python notebook from the "New" menu at the top-right:
# 
# <img src="newnotebook.png">
# 
# You should now be able to run Python in your browser!

# ## Time to build a neural network!
# First let's import some prerequisites

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,7) # Make the figures a bit bigger

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils


# ## Load training data

# In[3]:


nb_classes = 10

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("X_train original shape", X_train.shape)
print("y_train original shape", y_train.shape)


# Let's look at some examples of the training data

# In[4]:


for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[i], cmap='gray', interpolation='none')
    plt.title("Class {}".format(y_train[i]))

plt.show()

# ## Format the data for training
# Our neural-network is going to take a single vector for each training example, so we need to reshape the input so that each 28x28 image becomes a single 784 dimensional vector. We'll also scale the inputs to be in the range [0-1] rather than [0-255]

# In[5]:


X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)


# Modify the target matrices to be in the one-hot format, i.e.
# 
# ```
# 0 -> [1, 0, 0, 0, 0, 0, 0, 0, 0]
# 1 -> [0, 1, 0, 0, 0, 0, 0, 0, 0]
# 2 -> [0, 0, 1, 0, 0, 0, 0, 0, 0]
# etc.
# ```

# In[6]:


Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


# # Build the neural network
# Build the neural-network. Here we'll do a simple 3 layer fully connected network.
# <img src="figure.png" />

# In[7]:


model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu')) # An "activation" is just a non-linear function applied to the output
                              # of the layer above. Here, with a "rectified linear unit",
                              # we clamp all values below 0 to 0.
                           
model.add(Dropout(0.2))   # Dropout helps protect the model from memorizing or "overfitting" the training data
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax')) # This special "softmax" activation among other things,
                                 # ensures the output is a valid probaility distribution, that is
                                 # that its values are all non-negative and sum to 1.


# ## Compile the model
# Keras is built on top of Theano (and now TensorFlow as well), both packages that allow you to define a *computation graph* in Python, which they then compile and run efficiently on the CPU or GPU without the overhead of the Python interpreter.
# 
# When compiing a model, Keras asks you to specify your **loss function** and your **optimizer**. The loss function we'll use here is called *categorical crossentropy*, and is a loss function well-suited to comparing two probability distributions.
# 
# Here our predictions are probability distributions across the ten different digits (e.g. "we're 80% confident this image is a 3, 10% sure it's an 8, 5% it's a 2, etc."), and the target is a probability distribution with 100% for the correct category, and 0 for everything else. The cross-entropy is a measure of how different your predicted distribution is from the target distribution. [More detail at Wikipedia](https://en.wikipedia.org/wiki/Cross_entropy)
# 
# The optimizer helps determine how quickly the model learns, how resistent it is to getting "stuck" or "blowing up". We won't discuss this in too much detail, but "adam" is often a good choice (developed here at U of T).

# In[8]:


model.compile(loss='categorical_crossentropy', optimizer='adam')


# ## Train the model!
# This is the fun part: you can feed the training data loaded in earlier into this model and it will learn to classify digits

# In[9]:


model.fit(X_train, Y_train,
          batch_size=128, epochs=4,
          verbose=1,
          validation_data=(X_test, Y_test))


# ## Finally, evaluate its performance

# In[10]:


score = model.evaluate(X_test, Y_test,
                       verbose=0)
print('Test score:', score)
# print('Test accuracy:', score[1])


# ### Inspecting the output
# 
# It's always a good idea to inspect the output and make sure everything looks sane. Here we'll look at some examples it gets right, and some examples it gets wrong.

# In[11]:


# The predict_classes function outputs the highest probability class
# according to the trained classifier for each input example.
predicted_classes = np.argmax(model.predict(X_test), axis=-1)

# Check which items we got right / wrong
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]


# In[12]:


plt.figure()
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))

plt.figure()
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))

plt.show()

# # That's all!

# There are lots of other great examples at the Keras homepage at http://keras.io and in the source code at https://github.com/fchollet/keras
