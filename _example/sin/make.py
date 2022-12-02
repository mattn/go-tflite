import numpy as np
import tensorflow as tf

data = np.loadtxt('sin.csv', delimiter=',', unpack=True)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(30, input_shape=(1, )),
    tf.keras.layers.Activation('sigmoid'),
    tf.keras.layers.Dense(40),
    tf.keras.layers.Activation('sigmoid'),
    tf.keras.layers.Dense(1),
])
model.compile(
    loss='mean_squared_error',
    optimizer=tf.keras.optimizers.Adam(lr=0.1))
model.fit(data[0], data[1], epochs=1000, batch_size=20, verbose=0)
model.save('sin_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('sin_model.tflite', 'wb') as f:
    f.write(tflite_model)
