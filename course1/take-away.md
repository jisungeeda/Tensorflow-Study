# Basics
* 모델 구조 정의
```
model = tf.keras.Sequential([
	tf.keras.layers.Dense(units=1, input_shape=[1])
])
```
* 모델의 Loss function 과 optimizer 정의
```
model.compile(optimizer='sgd', loss='mean_squared_error')
```
* 모델 학습
```
model.filt(xs, ys, epochs=500)
```
* 모델 예측 (모델 예측은 compile 하지 않은 모델도 실행 가능)
```
model.predict([a])
```

# class keras.callbacks.Callback
* a method named 'on\_epoch\_end' gets called when an epoch finishes.
* fit 함수에 'callbacks' 라는 argument 를 넣을 수 있는데 이는 우리가 어떠한 metric 이 threshold 값에 도달했을 시에 학습을 멈추는 등의 제어 역할을 담당한다. 기능이 더 다양할 것 같은데 추후에 확인해야할 것 같다!
* 사용 예
```
class myCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if (logs.get('loss') <= 0.35):
			print('nloss: {}\ncancelling training'.format(logs.get('loss)))
			self.model.stop_training = True
callbacks = myCallback() 
history = model.fit(xs, ys, epochs=10, callbacks=callbacks)
```

# Training Tips
* 학습 history 정보는 fit 함수의 리턴 변수에 담긴다. print(history.history) 하면 log 볼 수 있다. Q) history 를 디스크에 저장하는 방법?
* 인풋은 normalize 해서 넣어주면 학습이 잘된다.
	* maxval = np.ndarray.max(train\_images.reshape(-1))
	* train\_images = train\_images / maxval
* predict 와 evaluate 차이: predict 는 라벨의 필요 없이, inference 한다. evaluate 은 라벨이 필요하고, inference 한 값과 비교하여 loss function 과 그 외 metric 을 계산한다. 따라서 evaluate 은 validation 시에 쓰이고 실제 사용될 때에는 predict 가 사용된다. 만약에 prediction 용으로 카피한 모델을 evaluate 하고 싶다면 model.compile 로 loss function과 optionally metric 을 정의해주어야 한다.

# class ImageGenerator (week4)
* Generate batches of tensor image data with real-time data augmentation.
```
train_datagen = ImageDataGenerator(rescale=1/255)
```
* Flow training images in batches of 128 using train\_datagen generator
```
train_generator = train_datagen.flow_from_directory(
    local_train_target,
    target_size=(150, 150),
    batch_size=128,
    class_mode='binary'
)
```
* insert a validation process during training
```
model.fit(
	train_generator,
	steps_per_epoch=8,
	epochs=15,
	verbose=1,
	validation_data=val_generator,
	validation_steps=2
)
```

# Model
* 모델은 레이어들의 집합이다. 이미 output 이 정해져있는 모델에서, 중간 레이어들의 output 을 찍어볼 수 없다. 중간 레이어들의 output 으로 정의되어있는 모델을 만들어야 한다. 만약 다음과 같이 정의되고 학습된 모델이 있다고 하자.

```
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
```
* 각 레이어들의 아웃풋을 찍어보고 싶다면, 
```
layer_outputs = [layer.output for layer in model.layers]
model_copied = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
```
한 뒤에 predict 메소드를 사용하면 된다.
* 모델을 Sequential 로 정의하는 방법과, Model 로 functional 하게 정의하는 방법이 있다.
* 모델을 functional 하게 정의하면 좋은 점:
	1. multiple inputs or outputs 가능
	2. skip connection 과 같은 sequential 하지 않은 레이어 처리 가능
	3. 레이어의 모든 층 확인 가능
* 모델을 그대로 카피하려면,
```
model_clone = tf.keras.models.clone_model(model)
```
* transfer\_learning 을 하기 위해 일부만 카피하려면, tf.keras.models.Model 을 사용하여 inputs 와 outputs 를 적절히 넣어주면 된다.
* What is a Convolution? A technique to isolate feats in images
* What is a Pooling? A technique to reduce the info in an img while maintaining feats
