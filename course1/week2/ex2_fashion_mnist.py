### Fashion MNIST ###
#%%
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
#%%
fashion_mnist = keras.datasets.fashion_mnist
#%%
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#%%
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#%% Data Exploration
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)
print(test_labels)
#%% Data Exploration
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

plt.show()
#%% Data Exploration
maxval = np.ndarray.max(train_images.reshape(-1))
minval = np.ndarray.min(train_images.reshape(-1))
print(maxval, minval)
maxval_label = np.ndarray.max(train_labels.reshape(-1))
minval_label = np.ndarray.min(train_labels.reshape(-1))
print(maxval_label, minval_label)
#%% Data Pre-processing
train_images = train_images / maxval
test_images = test_images / maxval
#%%
from utils import myCallback
#class myCallback(keras.callbacks.Callback):
#    def on_epoch_end(self, epoch, logs={}):
#        if (logs.get('loss') < 0.6):
#            print('loss is low so cancelling training!')
#            self.model.stop_training = True
callbacks = myCallback()
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(units=512, activation='relu'),
    keras.layers.Dense(units=10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
#%% Train the Model
history = model.fit(train_images, train_labels, epochs=30, callbacks=[callbacks])

#%% Predict with the trained Model
print(history.history)
print(history.epoch)

