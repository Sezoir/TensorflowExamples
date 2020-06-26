import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
klr = keras.layers

IMGSIZE = 160
training = False

def format_example(image, label):
    """
    returns an image that is reshaped to IMG_SIZE
    """
    # Cast to float as we want range between [-1,1] for better prediction
    image = tf.cast(image, tf.float32)
    # Note we are using Standardisation here for the model
    image = (image / 127.5) - 1
    # Resize image
    image = tf.image.resize(image, (IMGSIZE, IMGSIZE))
    return image, label

def run():
    (rawTrain, rawValidation, rawTest), metadata = tfds.load(
        'cats_vs_dogs',
        split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        with_info=True,
        as_supervised=True,
    )

    getLabelName = metadata.features['label'].int2str

    # Show some figures
    # for image, label in rawTest.take(2):
    #     plt.figure()
    #     plt.imshow(image)
    #     plt.title(get_label_name(label))
    # for image, label in rawValidation.take(2):
    #     plt.figure()
    #     plt.imshow(image)
    #     plt.title(get_label_name(label))
    # plt.show()

    # Resize all data
    train = rawTrain.map(format_example)
    test = rawTest.map(format_example)
    validation = rawValidation.map(format_example)

    # Shuffle and batch the images
    BATCH_SIZE = 32
    SHUFFLE_BUFFER_SIZE = 1000

    trainBatches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    validationBatches = validation.batch(BATCH_SIZE)
    testBatches = test.batch(BATCH_SIZE)

    # Define the shape of our images
    IMGSHAPE = (IMGSIZE, IMGSIZE, 3)

    # Create the base model of an existing model
    baseModel = keras.applications.MobileNetV2(input_shape=IMGSHAPE, include_top=False, weights='imagenet')

    # Disable training of the base model as we want to use the weights/bias that has already been trained
    baseModel.trainable = False

    # Get the average of each 2D feature map, and return a single 1280 element vector for each filter
    globalAverageLayer = klr.GlobalAveragePooling2D()
    # Create 1 dense node
    predictionLayer = klr.Dense(1)
    # Create our model
    model = keras.Sequential([
        baseModel,
        globalAverageLayer,
        predictionLayer
    ])
    # This affects how much the model is allowed to modify the weights/bias of the network
    # Made it low as we don't want to make to many changes to the model, as we are already using the baseModel
    baseLearningRate = 0.0001
    model.compile(optimizer=keras.optimizers.RMSprop(lr=baseLearningRate),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    # Get the initial accuracy with the pretrained model
    initialEpochs = 3
    validationSteps = 20
    # Check whether we want to train our dataset
    if(training == True):
        loss0, accuracy0 = model.evaluate(validationBatches, steps=validationSteps)

        # Train the model
        history = model.fit(trainBatches, epochs=initialEpochs, validation_data=validationBatches)
        acc = history.history["accuracy"]
        print(acc)

        # Now save and load the model
        model.save("dogs_vs_cats.h5")
    newModel = keras.models.load_model("dogs_vs_cats.h5")

    # Test the model
    predictions = newModel.predict(testBatches, batch_size=BATCH_SIZE, verbose=1)

    # Takes a prediction and turns it in the animal representation string literal
    def animalName(prediction):
        if prediction > 0:
            return "dog"
        else:
            return "cat"

    # Iterate through each test image, and print out what the NN thinks it is compared to what it actually is
    i = 0
    for image, label in test.take(10):
        print("We predict: ", animalName(predictions[i]), "It actually is: ", getLabelName(label))
        i = i+1



    return
