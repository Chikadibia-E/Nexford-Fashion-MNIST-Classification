#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the requird modules:

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import models, layers


# # Loading the Fashion MNIST Dataset

# In[2]:


# Load the dataset
data = fashion_mnist.load_data()


# # Data Exloration:

# In[3]:


# Data overview:
data


# In[4]:


# Spliting the data to test and train sets:

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


# In[5]:


# Display the shape of the dataset:

print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')


# In[6]:


# checking the max value of X_train, X_test set:

print(f'X_train max: {np.max(X_train)}')
print(f'X_test max: {np.max(X_test)}')


# In[7]:


y_train


# In[8]:


y = np.unique(y_train)


# In[9]:


y


# In[10]:


# Fashion classes in data original dataset:

fashion_classes = ['T-shirt/top',
'Trouser',
'Pullover',
'Dress',
'Coat',
'Sandal',
'Shirt',
'Sneaker',
'Bag',
'Ankle boot']


# In[11]:


# Create a dictionary to map y_train values to fashion classes
mapping = {i: fashion_classes[i] for i in y_train}

# Apply the mapping to y_train
fashion_labels = [mapping[val] for val in y_train]


# In[12]:


# Display any image and its label before normalization:

plt.figure()
plt.imshow(X_train[1])
plt.title(f'Label: {fashion_labels[1]}')
plt.colorbar()
plt.show()


# # Normalization:

# In[13]:


# Deep learning model works best with values between 0 and 1; Rescaling max value to 1.

norm_X_train = X_train/255
norm_X_test = X_test/255

# rechecking the max value of X_train, X_test set after normalizing:

print(f'Normalized X_train max: {np.max(norm_X_train)}')
print(f'Normalized X_test max: {np.max(norm_X_test)}')


# In[14]:


# Reshape the data to add a channel dimension
norm_X_train = norm_X_train.reshape((norm_X_train.shape[0], 28, 28, 1))
norm_X_test = norm_X_test.reshape((norm_X_test.shape[0], 28, 28, 1))


# In[15]:


# Display any image and its label after normalization:
plt.figure()
plt.imshow(norm_X_train[1])
plt.title(f'Label: {fashion_labels[1]}')
plt.colorbar()
plt.show()


# # Building the CNN model:

# In[16]:


# Building the CNN model:

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()


# # Compiling the model:

# In[17]:


# Compiling the model:

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# # Training the model:

# In[18]:


# Training the model:

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))


# # Predictions:

# In[19]:


# Predictions:

predictions = model.predict(norm_X_test[:2])
print(predictions)


# In[ ]:




