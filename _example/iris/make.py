import pandas as pd
import tensorflow as tf

names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Name']
train = pd.read_csv('iris.csv', names=names, header=1)
xx, spacies = train, train.pop('Name')
yy = pd.get_dummies(spacies)
dataset = tf.data.Dataset.from_tensor_slices((xx.values, yy))
dataset = dataset.batch(32).shuffle(1000).repeat()

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, input_dim=4),
    tf.keras.layers.Dense(3, activation=tf.nn.softmax),
])

model.summary()
model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy'])
model.fit(dataset, steps_per_epoch=32, epochs=100, verbose=1)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("iris.tflite", "wb") as f:
    f.write(tflite_model)
