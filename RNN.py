import numpy as np
import os

import tensorflow as tf
from tensorflow import keras
dts = keras.datasets
klr = keras.layers
prp = keras.preprocessing

# Note here I am using a Recurrent Neural Network for modeling sequence data such as time series or natural language.

# Do we train the model, or use pre-trained model
training = True

def run():
    VOCAB_SIZE = 88584

    MAXLEN = 250
    BATCH_SIZE = 64

    # This dataset features a bunch of reviews of movies, with the label dictation whether the review is positive or negative.
    # Note that the words have already been encoded to integers
    (trainData, trainLabels), (testData, testLabels) = dts.imdb.load_data(num_words=VOCAB_SIZE)

    # NN only accept inputs of the same length, so we left pad the strings with 0's and cut out any characters past MAXLEN.
    trainData = prp.sequence.pad_sequences(trainData, MAXLEN)
    testData = prp.sequence.pad_sequences(testData, MAXLEN)

    if(training):
        model = keras.Sequential([
            klr.Embedding(VOCAB_SIZE, 32),
            klr.LSTM(32),
            klr.Dense(1, activation="sigmoid")
        ])

        model.summary()

        model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['acc'])

        history = model.fit(trainData, trainLabels, epochs=10, validation_split=0.2)

        model.save("RNN.h5")

    newModel = keras.models.load_model("RNN.h5")

    results = newModel.evaluate(testData, testLabels)
    print(results)

    # Get the word index used in the dataset
    word_index = dts.imdb.get_word_index()
    # Encode a given sentence into the same format for the imdb dataset
    def encode_text(text):
        # Split the sentence in "tokens" which is every word in the text
        tokens = keras.preprocessing.text.text_to_word_sequence(text)
        # Encode the tokens
        tokens = [word_index[word] if word in word_index else 0 for word in tokens]
        # Return the encoded text, but making sure the input is the same as we used in our NN
        # Note that pad_sequences only works on a list of sequences, so we wrap our list in another list and then take
        # the first list
        return prp.sequence.pad_sequences([tokens], MAXLEN)[0]

    # Test text
    text = "that movie was just amazing, so amazing"
    # Encode text
    encoded = encode_text(text)
    print(encoded)

    # Simply the inverse of the encode_text
    reverse_word_index = {value: key for (key, value) in word_index.items()}
    def decode_integers(integers):
        PAD = 0
        text = ""
        for num in integers:
            if num != PAD:
                text += reverse_word_index[num] + " "

        return text[:-1]
    print(decode_integers(encoded))

    # now time to make a prediction

    def predict(text):
        # Encode the text
        encoded_text = encode_text(text)
        # We create an array of 0 in the same shape as what our NN expects (expects (:,250)) for 1 prediction
        pred = np.zeros((1, 250))
        # Insert our encoded text in the array
        pred[0] = encoded_text
        # Make the prediction
        result = newModel.predict(pred)
        print(result[0])

    positive_review = "That movie was! really loved it and would great watch it again because it was amazingly great"
    predict(positive_review)

    negative_review = "that movie really sucked. I hated it and wouldn't watch it again. Was one of the worst things I've ever watched"
    predict(negative_review)




    return