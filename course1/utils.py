import tensorflow as tf
from tensorflow import keras

class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') >= 0.93):
            print('accuracy: {}\n Cancelling training!'.format(logs.get('accuracy')))
            self.model.stop_training = True
