import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import numpy as np

class KerasLSTM:

    def generate(self, hidden_size, timesteps, features, output_dim, layers):
        model = Sequential()
        # 50 for number of timesteps, 3 for features
        model.add(LSTM(hidden_size, input_shape=(timesteps, features)))
        # dense takes in output dimensionality
        model.add(Dense(output_dim))
        # add softmax activation
        model.add(Activation('softmax'))
        # indicate loss and optimizer
        model.compile(loss='categorical_crossentropy', optimizer='sgd')

        return model

    def train(self, model, train_x, train_y, test_x, test_y, epochs):
        # train the model
        # validation_data=(test_x, test_y),
        fit = model.fit(train_x, train_y, epochs=epochs, verbose=2, shuffle=False)
        # return trained
        return fit

    def test():
