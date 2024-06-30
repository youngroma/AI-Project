import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import utils

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

class_names = ['T-shir/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal','Shirt', 'Sneaker', 'Bag', 'Ankle boot']

x_train = x_train / 255
x_test = x_test / 255

plt.figure()
plt.imshow(x_train[0])
plt.colorbar()
plt.grid(False)
# plt.show()

# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, +1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(x_train[i])
#     plt.xlabel(class_names[y_train[i]])
#     plt.show()


#AI
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer=tf.keras.optimizers.SGD(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

#Teaching AI
model.fit(x_train, y_train, epochs=10)

#check accuracy
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy', test_acc)

#predictions
predictions = model.predict(x_train)
print(predictions[0])
print(np.argmax(predictions[0]))
print(y_train[0])

plt.figure()
plt.imshow(x_train[15])
plt.colorbar()
plt.grid(False)
plt.show()

# class_names[np.argmax(predictions[12])]
