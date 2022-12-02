import numpy as np
import tensorflow as tf

x_train = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
]).astype(np.float32)

y_train = np.array([
    0,
    1,
    1,
    0,
]).astype(np.float32)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1),
])

model.summary()
model.compile(optimizer='adam', loss="mse", metrics=["mae"])
model.fit(x_train, y_train, epochs=500, batch_size=4)


def representative_dataset():
    for i in range(4):
        yield [x_train[i]]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32
converter.representative_dataset = representative_dataset
model_tflite = converter.convert()
with open('xor_model.tflite', 'wb') as f:
    f.write(model_tflite)
