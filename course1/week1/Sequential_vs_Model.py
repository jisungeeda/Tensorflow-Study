### keras.Sequential vs. keras.Model ###
#%% keras.Sequential
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

#%% keras.Model
inputs = tf.keras.Input(shape=(28, 28, 1, ))
x1 = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(inputs)
x1p = tf.keras.layers.MaxPooling2D(2, 2)(x1)
x2 = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(x1p)
x2p = tf.keras.layers.MaxPooling2D(2, 2)(x2)
x_flat = tf.keras.layers.Flatten()(x2p)
x_lin = tf.keras.layers.Dense(128, activation='relu')(x_flat)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x_lin)

model2 = tf.keras.Model(inputs=inputs, outputs=[x1, x1p, x2, x2p, outputs])
model2.summary()

#%%
print(model.layers[-1].output)
#%%
m = tf.keras.models.clone_model(model)
print(m.summary())

### Takeaway ###
# Sequential 로 모델을 정의할 때와 Model 로 정의할 때와 summary() 아웃풋이 다르다. Sequential 로 만들면 input layer 가 명시되지 않는데, Model 로 정의하면 input layer 가 명시되어있다.. 이유는 모르겠다.
