import math

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
img = keras.preprocessing.image

def run():

    datagen = img.ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    (xTrain, yTrain), (xTest, yTest) = keras.datasets.cifar10.load_data()

    # Bias is typically between 0-1, so want to scale input to be between 0-1 to help the model
    xTrain, xTest = xTrain / 255.0, xTest / 255.0

    testImg = xTrain[20]
    image = img.img_to_array(testImg)
    print(image.shape)
    image = image.reshape((1,) + image.shape)

    i = 0
    fig, axs = plt.subplots(2, 2)
    for batch in datagen.flow(image, save_prefix='test', save_format='jpeg'):
        # axs.figure(i)
        axs[int(math.floor(i/2)), int(math.fmod(i, 2))].imshow(img.img_to_array(batch[0]))
        i += 1
        if i > 3:  # show 4 images
            break
    plt.show()

    return