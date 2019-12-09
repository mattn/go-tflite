import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.lite.python import lite

X_train = np.array([[0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0]])
Y_train = np.array([0.0,
                    1.0,
                    1.0,
                    0.0])
model = Sequential()
output_count_layer0 = 2
model.add(
    Dense(
      output_count_layer0,
      input_shape=(2, ),
      activation='sigmoid'))  # Need to specify input shape for input layer
output_count_layer1 = 1
model.add(Dense(output_count_layer1, activation='linear'))
model.compile(
    loss='mean_squared_error', optimizer=RMSprop(), metrics=['accuracy'])
BATCH_SIZE = 4
history = model.fit(
    X_train, Y_train, batch_size=BATCH_SIZE, epochs=3600, verbose=1)
X_test = X_train
Y_test = Y_train
score = model.evaluate(X_test, Y_test, verbose=0)
model.save('xor_model.h5')

converter = lite.TFLiteConverter.from_keras_model_file('xor_model.h5')
tflite_model = converter.convert()
open('public/xor_model.tflite', 'wb').write(tflite_model)
