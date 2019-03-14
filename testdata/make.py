import numpy as np 
from tensorflow.contrib.keras.api.keras.models import Sequential, model_from_json
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Activation
from tensorflow.contrib.keras.api.keras.optimizers import SGD
import tensorflow.contrib.lite as lite

model = Sequential()
model.add(Dense(8, input_dim = 2))
model.add(Activation('tanh'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = SGD(lr = 0.1))
model.fit(
    np.array([[0, 0], [0, 1.0], [1.0, 0], [1.0, 1.0]]),
    np.array([[0.0], [1.0], [1.0], [0.0]]),
    batch_size = 1, epochs = 300)
model.save('xor_model.h5')

converter = lite.TFLiteConverter.from_keras_model_file("xor_model.h5")
tflite_model = converter.convert()
open("xor_model.tflite", "wb").write(tflite_model)
