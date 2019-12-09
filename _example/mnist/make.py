from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.lite.python import lite
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation("softmax"))
model.compile(
    loss="categorical_crossentropy", optimizer=Adam(), metrics=['accuracy'])
model.fit(
    x_train, y_train, batch_size=128, epochs=20,
    verbose=1, validation_data=(x_test, y_test))
model.save('mnist_model.h5')

converter = lite.TFLiteConverter.from_keras_model_file('mnist_model.h5')
tflite_model = converter.convert()
open('mnist_model.tflite', 'wb').write(tflite_model)
