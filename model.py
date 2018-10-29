import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import numpy as np

class RNN:

    def __init__(self):
        """
        Initialize the Sequential Model
        """
        self.model = Sequential()

    def generate(self, hidden_size, timesteps, features, output_dim, layers):
        """
        Generate RNN defined
        """
        # 50 for number of timesteps, 3 for features
        self.model.add(LSTM(hidden_size, input_shape=(timesteps, features)))
        # dense takes in output dimensionality
        self.model.add(Dense(output_dim))
        # add softmax activation
        self.model.add(Activation('softmax'))
        # indicate loss and optimizer
        self.model.compile(loss='categorical_crossentropy', optimizer='sgd')

        return model

    def train(self, model, train_x, train_y, test_x, test_y, epochs):
        """
        Train the model
        """
        # train the model
        # validation_data=(test_x, test_y),
        history = self.model.fit(train_x, train_y, epochs=epochs, verbose=2, shuffle=False)
        
        return history

    def test():
