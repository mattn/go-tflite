import numpy as np
import tensorflow as tf

x = np.array([[0, 0], [0, 1.0], [1.0, 0], [1.0, 1.0]], dtype=np.float32)
y = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float32)

tf.random.set_seed(1)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(8, activation="tanh"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.SGD(learning_rate=0.5))
model.fit(x, y, batch_size=1, epochs=500, verbose=0)
print("predictions:", model.predict(x, verbose=0).ravel())

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("xor_model_sig.tflite", "wb").write(tflite_model)

interp = tf.lite.Interpreter(model_content=tflite_model)
print("signatures:", interp.get_signature_list())
