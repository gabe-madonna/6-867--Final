import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Bidirectional
from keras.layers import LSTM
from sklearn.neighbors import KNeighborsClassifier
from utils import *
import datetime
import numpy as np
import json
import pickle
from scipy.spatial.distance import cdist


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
            self.model.add(LSTM(hidden_size, input_shape=input_shape, return_sequences=True))

        # flatten output befor elast dense later
        self.model.add(Flatten())
        # dense takes in output dimensionality
        self.model.add(Dense(output_dim))
        # add softmax activation
        self.model.add(Activation('softmax'))
        # indicate loss and optimizer
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        print("===== FINISHED BUILDING MODEL ======")

        self.layers = layers
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.input_shape = input_shape

        return self.model

    def train(self, train_x, train_y, epochs=50):
        '''
        :param train_x:
        :param train_y:
        :param epochs:
        :return losses, accuracies: lists of performances by epoch
        '''
        # train the model
        # validation_data=(test_x, test_y),

        self.epochs = epochs

        print("===== TRAINING MODEL ======")

        history = self.model.fit(train_x, train_y, epochs=epochs, verbose=2, shuffle=False)

        print(self.model.summary())

        print("===== FINISHED TRAINING MODEL ======")
        for i in range(100):
            try:
                pickle.dump(self.model, open("models/model{0:04d}.p".format(i), "wb"))
            except:
                pass
        losses, accuracies = history.history['loss'], history.history['acc']
        return losses, accuracies

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
        total_unique, total_counts = np.unique(nums, return_counts=True)
        unique, counts = np.unique(misses, return_counts=True)
        unique = [self.NUM2LET[u] for u in unique]
        total_unique = [self.NUM2LET[u] for u in total_unique]
        miss_dict = dict(zip(unique, counts))
        total_dict = dict(zip(total_unique, total_counts))
        error_dict = {k: miss_dict[k]/total_dict[k] for k in miss_dict.keys()}
        # print('miss_dict:', miss_dict)
        # print('totals_dict:', total_dict)
        # print('error_dict:', error_dict)
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

        print("===== FINISHED TESTING MODEL ======")

        # print('Test Accuracy: %.3f' % totalAccuracy)

        with open("results.txt", "a") as myfile:
            myfile.write("-------------------\n")
            myfile.write("RNN\n")
            myfile.write(str(datetime.datetime.now()) + '\nTesting loss: {}, acc: {}\n'.format(loss, acc))
            myfile.write("HYPERPARAMS: ")
            myfile.write('Layers: {}, Hidden Size: {}, Output Dim: {}, Epochs: {}\n'.format(self.layers, self.hidden_size, self.output_dim, self.epochs))
            myfile.write('Misclassified files: {}\n'.format(miss_dict))

        print('nice work, Pramoda')

        return loss, acc, error_dict


class CNN:

    def __init__(self):
        '''
        initialize the Sequential Model
        '''
        self.model = Sequential()
        self.layers = 2
        self.epochs = 0
        self.NUM2LET = None

    def generate(self, NUM2LET, hidden_size=50, output_dim=20, input_shape=(50,3), layers=1):
        '''
        generate the model
        '''
        self.NUM2LET = NUM2LET
        for i in range(layers):
            if i == 0:
                self.model.add(Conv2D(hidden_size, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='same'))
            else:
                self.model.add(Conv2D(hidden_size, kernel_size=(3, 3), activation='relu', padding='same'))
            # self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        # self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(100, activation='relu'))
        # 20 for number of classes
        self.model.add(Dense(output_dim, activation='softmax'))
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

        print(self.model.summary())

        print("===== FINISHED TRAINING MODEL ======")
        
        losses, accuracies = history.history['loss'], history.history['acc']
        return losses, accuracies

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
        total_unique, total_counts = np.unique(nums, return_counts=True)
        unique, counts = np.unique(misses, return_counts=True)
        unique = [self.NUM2LET[u] for u in unique]
        total_unique = [self.NUM2LET[u] for u in total_unique]
        miss_dict = dict(zip(unique, counts))
        total_dict = dict(zip(total_unique, total_counts))
        error_dict = {k: miss_dict[k]/total_dict[k] for k in miss_dict.keys()}
        # print('miss_dict:', miss_dict)
        # print('totals_dict:', total_dict)
        # print('error_dict:', error_dict)
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

        print("===== FINISHED TESTING MODEL ======")

        # print('Test Accuracy: %.3f' % totalAccuracy)

        # with open("results.txt", "a") as myfile:
        #     myfile.write("-------------------\n")
        #     myfile.write("RNN\n")
        #     myfile.write(str(datetime.datetime.now()) + '\nTesting loss: {}, acc: {}\n'.format(loss, acc))
        #     myfile.write("HYPERPARAMS: ")
        #     myfile.write('Layers: {}, Hidden Size: {}, Output Dim: {}, Epochs: {}\n'.format(self.layers, self.hidden_size, self.output_dim, self.epochs))
        #     myfile.write('Misclassified files: {}\n'.format(miss_dict))

        print('nice work, Pramoda')

        return loss, acc, error_dict


class KNN():

    def __init__(self, numNeighbors, NUM2LET):
        self.numNeighbors = numNeighbors
        self.model = KNeighborsClassifier(n_neighbors=self.numNeighbors)
        self.NUM2LET = NUM2LET

    def train(self, train_X, train_Y):
        self.model.fit(train_X, train_Y)

    def getScore(self, test_X, test_Y):
        return self.model.score(test_X, test_Y)

    def modelPredict(self, test_X):
        return self.model.predict(test_X)

    def get_stats(self, test_X, test_Y):
        '''
        get accuracy, error dict
        :return:
        '''

        y_hat = self.modelPredict(test_X)
        y_hat = np.array([np.argmax(y_hat[i]) for i in range(len(y_hat))])
        incorrects = [y_hat[i] != np.argmax(test_Y[i]) for i in range(len(y_hat))]
        # print(sum(incorrects))
        accuracy = sum(incorrects) / len(incorrects)
        nums = np.array([np.argmax(yi) for yi in test_Y])
        misses = nums[incorrects]
        total_unique, total_counts = np.unique(nums, return_counts=True)
        unique, counts = np.unique(misses, return_counts=True)
        unique = [self.NUM2LET[u] for u in unique]
        total_unique = [self.NUM2LET[u] for u in total_unique]
        miss_dict = dict(zip(unique, counts))
        total_dict = dict(zip(total_unique, total_counts))
        error_dict = {k: miss_dict[k] / total_dict[k] for k in miss_dict.keys()}
        # print('miss_dict:', miss_dict)
        # print('totals_dict:', total_dict)
        # print('error_dict:', error_dict)
        print('missed {}/{}'.format(len(misses), len(y_hat)))
        return accuracy, error_dict


class Template:

    def __init__(self):
        self.letters = []
        self.averages = []
        self.averages_dict = []
        self.miss_dict = {}

    def average_letters(self, letters_train):
        '''
        generate average matrices for each letter
        :param letters (dict): dictionary of letter to list of matrices
        :return keys, avgs, averages: letters, averages, and average dict
        '''
        averages = {}
        keys, avgs = [], []

        for letter in letters_train:
            samples = letters_train[letter]
            avg = np.mean(samples, axis=0)
            averages[letter] = avg
            keys.append(letter)
            avgs.append(avg)

        self.letters = keys
        self.averages = avgs
        self.averages_dict = averages
        
        return keys, avgs, averages

    def find_closest(self, sample, distance_metric):
        '''
        find the index of the closest matrix using distance_metric
        :param sample (np.array): number of the dataset
        '''
        dist = []
        for avg in self.averages:
            dist.append(np.average(cdist(sample, avg, metric=distance_metric)))

        ind = np.argmin(dist)

        return ind, dist

    def test_letters(self, test_X, test_y, num2let, distance_metric):
        predicted = []
        accuracy = 0
        for index, x in enumerate(test_X):
            ind, dist = self.find_closest(x, distance_metric)
            correct_ind = np.argmax(test_y[index])
            predicted.append((self.letters[ind], num2let[correct_ind]))
            if self.letters[ind] == num2let[correct_ind]:
                accuracy += 1
            else:
                if num2let[correct_ind] in self.miss_dict:
                    self.miss_dict[num2let[correct_ind]] += 1
                else:
                    self.miss_dict[num2let[correct_ind]] = 1

        accuracy /= len(test_y)

        nums = np.array([np.argmax(yi) for yi in test_y])

        total_unique, total_counts = np.unique(nums, return_counts=True)

        total_unique = [num2let[u] for u in total_unique]

        total_dict = dict(zip(total_unique, total_counts))
        error_dict = {k: self.miss_dict[k] / total_dict[k] for k in self.miss_dict.keys()}
        print('miss_dict:', self.miss_dict)
        print('totals_dict:', total_dict)
        print('error_dict:', error_dict)

        return accuracy, error_dict


if __name__ == '__main__':
    pass
    
    # rnn = RNN()
    # model = rnn.generate(100, i50, 3, 20, 10)
    # print(model)
