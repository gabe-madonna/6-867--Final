import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Bidirectional
from keras.layers import LSTM
from utils import *

import datetime
import numpy as np
import json

class RNN:

    def __init__(self):
        '''
        initialize the Sequential Model
        '''
        self.model = Sequential()
        self.layers = 1
        self.output_dim = 20
        self.hidden_size = 50
        self.input_shape = (50, 3)
        self.epochs = 50
        self.NUM2LET = {}

    def generate(self, NUM2LET, hidden_size=50, output_dim=20, input_shape=(50,3), layers=1):
        '''
        generate RNN defined
        :hidden_size (int): number of units in each hidden layer (LSTM)
        :timesteps (int): number of timesteps per sample
        :features (int): number of features each sample has
        :output_dim (int): number of dimensions in the output i.e. number of classes
        :layers (int): number of layers
        '''
        self.NUM2LET = NUM2LET
        print("===== BUILDING MODEL ======")

        # 50 for number of timesteps, 3 for features
        for i in range(layers):
            self.model.add(Bidirectional(LSTM(hidden_size, input_shape=input_shape, return_sequences=True)))
        # flatten output befor elast dense later
        self.model.add(Flatten())
        # dense takes in output dimensionality
        self.model.add(Dense(output_dim))
        # add softmax activation
        self.model.add(Activation('softmax'))
        # indicate loss and optimizer
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        print("===== FINISHED BUILDING MODEL ======")
        print(self.model.summary())

        self.layers = layers
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.input_shape = input_shape

        return self.model

    def train(self, train_x, train_y, epochs=50):
        '''
        train the model
        :train_x (np.array): inputs (x)
        :train_y (np.array): outputs (y)
        '''
        # train the model
        # validation_data=(test_x, test_y),

        self.epochs = epochs

        print("===== TRAINING MODEL ======")

        history = self.model.fit(train_x, train_y, epochs=epochs, verbose=2, shuffle=False)

        print("===== FINISHED TRAINING MODEL ======")
        
        return history

    def test(self, test_X, test_Y):
        '''
        test the model
        '''
        # make a prediction
        print("===== TESTING MODEL ======")
        loss, acc = self.model.evaluate(test_X, test_Y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

        y_hat = self.model.predict(test_X)
        y_hat = np.array([np.argmax(y_hat[i]) for i in range(len(y_hat))])
        incorrects = [y_hat[i] != np.argmax(test_Y[i]) for i in range(len(y_hat))]
        print(sum(incorrects))
        nums = np.array([np.argmax(yi) for yi in test_Y])
        misses = nums[incorrects]
        unique, counts = np.unique(misses, return_counts=True)
        unique = [self.NUM2LET[u] for u in unique]
        miss_dict = dict(zip(unique, counts))
        print(miss_dict)
        print('missed {}/{}'.format(len(misses), len(y_hat)))

        # yhat = self.model.predict(test_X, verbose=1)
        # # print(yhat)
        # # test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
        # # calculate RMSE
        # totalAccuracy = 0.0
        # for i in range(len(test_Y)):
        #     if np.argmax(test_Y[i]) == np.argmax(yhat[i]):
        #         totalAccuracy += 1
        # totalAccuracy/= len(test_Y)

        # print("===== FINISHED TESTING MODEL ======")

        # print('Test Accuracy: %.3f' % totalAccuracy)

        with open("results.txt", "a") as myfile:
            myfile.write("-------------------\n")
            myfile.write("RNN\n")
            myfile.write(str(datetime.datetime.now()) + '\nTesting loss: {}, acc: {}\n'.format(loss, acc))
            myfile.write("HYPERPARAMS: ")
            myfile.write('Layers: {}, Hidden Size: {}, Output Dim: {}, Epochs: {}\n'.format(self.layers, self.hidden_size, self.output_dim, self.epochs))
            myfile.write('Misclassified files: {}\n'.format(miss_dict))

        print('nice work, Pramoda')

class CNN:

    def __init__(self):
        '''
        initialize the Sequential Model
        '''
        self.model = Sequential()
        self.layers = 2
        self.epochs = 0

    def generate(self, input_shape):
        '''
        generate the model
        '''
        self.model.add(Conv2D(32, kernel_size=(3, 3), strides=2, activation='relu', input_shape=input_shape, padding='same'))
        # self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

        # self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(100, activation='relu'))
        # 20 for number of classes
        self.model.add(Dense(20, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return self.model

    def train(self, train_x, train_y, epochs=50):
        '''
        train the model
        :train_x (np.array): inputs (x)
        :train_y (np.array): outputs (y)
        '''
        # train the model
        # validation_data=(test_x, test_y),

        self.epochs = epochs

        print("===== TRAINING MODEL ======")

        history = self.model.fit(train_x, train_y, epochs=epochs, verbose=2, shuffle=False)

        print("===== FINISHED TRAINING MODEL ======")
        
        return history

    def test(self, test_X, test_Y):
        '''
        test the model
        '''
        # make a prediction
        print("===== TESTING MODEL ======")
        loss, acc = self.model.evaluate(test_X, test_Y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

        # get predicted output for data
        y_hat = self.model.predict(test_X)
        y_hat = np.array([np.argmax(y_hat[i]) for i in range(len(y_hat))])

        # find which characters were incorrectly classified
        incorrects = [y_hat[i] != np.argmax(test_Y[i]) for i in range(len(y_hat))]
        print(sum(incorrects))
        nums = np.array([np.argmax(yi) for yi in test_Y])
        misses = nums[incorrects]
        unique, counts = np.unique(misses, return_counts=True)
        unique = [self.NUM2LET[u+1] for u in unique]
        miss_dict = dict(zip(unique, counts))
        print(miss_dict)
        print('missed {}/{}'.format(len(misses), len(y_hat)))

        with open("results.txt", "a") as myfile:
            myfile.write("-------------------\n")
            myfile.write("CNN\n")
            myfile.write(str(datetime.datetime.now()) + '\nTesting loss: {}, acc: {}\n'.format(loss, acc))
            myfile.write("HYPERPARAMS: ")
            myfile.write('Layers: {}, Epochs: {}\n'.format(self.layers, self.epochs))
            myfile.write('Misclassified files: {}\n'.format(miss_dict))

        print('nice work, Pramoda')


if __name__ == '__main__':
    
    rnn = RNN()
    model = rnn.generate(100, 50, 3, 20, 10)
    print(model)