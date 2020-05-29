### CNN on Fashion MNIST ###
#%%
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

#%%
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#%%
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#%%
numimgs = train_images.shape[0]
sz = train_images.shape[1]
train_images = train_images.reshape(numimgs, sz, sz, 1) / 255.0
#test_images = test_images.reshape(test_images.shape[0], sz, sz, 1) / 255.0
#print(train_images.shape)
#%%

#tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

#%%
from utils import myCallback
callbacks = myCallback()
history = model.fit(train_images, train_labels, epochs=10, callbacks=[callbacks])

#%%
test_images = test_images / 255.0
test_loss = model.evaluate(test_images, test_labels)
#%%
print(test_loss)
#%%
print(model.summary())
print(model.input)
print(model.layers[0].output)
print(model.summary())
print('\n')
#for i in model.layers:
#    print('{} ||| {}'.format(i.input, i.output))
#%% Visualizing the Convolutions and Pooling
from tensorflow.keras import models
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
#activation_model = tf.keras.Model(inputs = model.input, outputs = layer_outputs)
print(activation_model.summary())
print(activation_model.input)
#%%
#f1 = activation_model.predict(test_images[0].reshape(1, 28, 28, 1))[0]
#print(f1.shape)
#print(f1[0, :, :, 0])
#f1 = model.predict(test_images[0].reshape(1, 28, 28, 1))
#print(model.layers[0].output)
#%%
#print(test_labels[:100])
#print(activation_model.evaluate(test_images[0].reshape(1,28,28,1), test_labels[0]))
#print(activation_model.predict(test_images[0].reshape(1,28,28,1)))
#test_labels[2]
#%% Visualizing the Convolutions and Pooling
import matplotlib.pyplot as plt
f, axarr = plt.subplots(3,4)
FIRST_IMAGE = 0
SECOND_IMAGE = 23
THIRD_IMAGE = 28
CONVOLUTION_NUMBER = 1
#print(activation_model.summary())
for x in range(0, 4):
    # First img
    f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[0, x].imshow(f1[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[0, x].grid(False)
    # Second img
    f1 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[1, x].imshow(f1[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[1, x].grid(False)
    # Third img
    f1 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[2, x].imshow(f1[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[2, x].grid(False)

#%%
#print(model.summary())
#activation_model.evaluate(test_images, test_labels)
#print(test_labels[0])
#print(test_labels[23])
#print(test_labels[28])
#test_loss = model.evaluate(test_images, test_labels)
#%%
#test_images = test_images.reshape(test_images.shape[0], sz, sz, 1) / 255.0