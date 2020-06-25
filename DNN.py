import numpy as np

import tensorflow as tf
from tensorflow import keras

# Note here I am using a Dense Neural Network, which works on a global scale,
# and so looks for patterns in a specific area of the image.

def run():
    mnist = tf.keras.datasets.fashion_mnist

    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print(xTrain.shape)
    print(xTest.shape)
    print(xTrain[0, 23, 23])
    print(yTrain[:10])

    classNames = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Bias is typically between 0-1, so want to scale input to be between 0-1 to help the model
    xTrain, xTest = xTrain / 255.0, xTest / 255.0

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),  # Input layer (1)
        keras.layers.Dense(128, activation="relu"),  # Hidden layer (2)
        keras.layers.Dense(10, activation="softmax")  # Output layer (3)
    ])

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(xTrain, yTrain, epochs=10)

    testLoss, testAcc = model.evaluate(xTest, yTest, verbose=1)
    print("Test accuracy: ", testAcc)

    predictions = model.predict(xTest)
    for x in range(10):
        print("We predict: ", np.argmax(predictions[x]), "It actually is: ", yTest[x])

    return