import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Sequential stack of layers, each after another
model = tf.keras.models.Sequential([
    # Flattens input
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # Dense neural network with x layers, activation function = Output-alignment
    tf.keras.layers.Dense(128, activation='relu'),
    # Dropout sets randomly inputs to 0, prevents overfitting
    tf.keras.layers.Dropout(0.2),
    # 10 Dense output layers
    tf.keras.layers.Dense(10)
])

# predict number of 1st image
predictions = model(x_train[:1]).numpy()
# Apply softmax to output, squish sum of predictions for numbers between 0 and 1
tf.nn.softmax(predictions).numpy()

# Crossentropy loss
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()

# Creation of model
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# Training of model
model.fit(x_train, y_train, epochs=5)

# Evaluation of accuracy based on test data
model.evaluate(x_test,  y_test, verbose=2)

# New model for probability
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

# Actual predictions
predictions = probability_model.predict(x_test)
# Take biggest arg and take the corresponding number
classes = np.argmax(predictions, axis=1)

print(classes)

