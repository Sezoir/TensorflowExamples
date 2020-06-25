import numpy as np

import tensorflow as tf
from tensorflow import keras
klr = keras.layers


# Note here I am using a Convolutional Neural Networks will find specific patterns anywhere in the image.
# Multiple Convolutional Layers act as filters to tell the NN what "feature" to look for.

def run():
    (xTrain, yTrain), (xTest, yTest) = keras.datasets.cifar10.load_data()

    # Bias is typically between 0-1, so want to scale input to be between 0-1 to help the model
    xTrain, xTest = xTrain / 255.0, xTest / 255.0

    classNames = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    model = keras.models.Sequential()
    # Create convolutional layer
    model.add(klr.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(klr.MaxPooling2D((2, 2))) # Apply pooling
    model.add(klr.Conv2D(64, (3, 3), activation='relu'))
    model.add(klr.MaxPooling2D((2, 2)))
    model.add(klr.Conv2D(64, (3, 3), activation='relu'))

    model.summary()

    # Now create the classifier
    model.add(klr.Flatten())
    model.add(klr.Dense(64, activation="relu"))
    model.add(klr.Dense(10))

    model.summary()

    model.compile(optimizer="adam",
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    history = model.fit(xTrain, yTrain, epochs=10, validation_data=(xTest, yTest))

    lossTest, accTest = model.evaluate(xTest, yTest, verbose=2)
    print("Test accuracy: ", accTest)

    predictions = model.predict(xTest)
    for x in range(10):
        print("We predict: ", np.argmax(predictions[x]), "It actually is: ", yTest[x])

    return
