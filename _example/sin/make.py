import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.lite.python import lite

data = np.loadtxt('sin.csv', delimiter=',', unpack=True)
model = Sequential()
model.add(Dense(30, input_shape=(1, )))
model.add(Activation('sigmoid'))
model.add(Dense(40))
model.add(Activation('sigmoid'))
model.add(Dense(1))
sgd = Adam(lr=0.1)
model.compile(loss='mean_squared_error', optimizer=sgd)
model.fit(data[0], data[1], epochs=1000, batch_size=20, verbose=0)
model.save('sin_model.h5')

converter = lite.TFLiteConverter.from_keras_model_file('sin_model.h5')
tflite_model = converter.convert()
open('sin_model.tflite', 'wb').write(tflite_model)
