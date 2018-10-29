import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.layers import LSTM

import numpy as np

class RNN:

    def __init__(self):
        '''
        initialize the Sequential Model
        '''
        self.model = Sequential()

    def generate(self, hidden_size, timesteps, features, output_dim, layers):
        '''
        generate RNN defined
        :hidden_size (int): number of units in each hidden layer (LSTM)
        :timesteps (int): number of timesteps per sample
        :features (int): number of features each sample has
        :output_dim (int): number of dimensions in the output i.e. number of classes
        :layers (int): number of layers
        '''
        # 50 for number of timesteps, 3 for features
        self.model.add(LSTM(hidden_size, input_shape=(timesteps, features)))
        # dense takes in output dimensionality
        self.model.add(Dense(output_dim))
        # add softmax activation
        self.model.add(Activation('softmax'))
        # indicate loss and optimizer
        self.model.compile(loss='categorical_crossentropy', optimizer='sgd')
        return self.model

    def train(self, model, train_x, train_y, epochs):
        '''
        train the model
        :train_x (np.array): inputs (x)
        :train_y (np.array): outputs (y)
        '''
        # train the model
        # validation_data=(test_x, test_y),
        history = self.model.fit(train_x, train_y, epochs=epochs, verbose=2, shuffle=False)
        
        return history

    def test(self, test_X, test_Y):
        '''
        test the model
        '''
        # make a prediction
        yhat = model.predict(test_X)
        test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
        # calculate RMSE
        totalAccuracy = 0.0
        for i in range(len(test_Y)):
            if test_Y[i] == yhat[i]:
                totalAccuracy += 1
        totalAccuracy/= len(test_Y)
        print('Test Accuracy: %.3f' % totalAccuracy)

class CNN:

    def __init__(self):
        '''
        initialize the Sequential Model
        '''
        self.model = Sequential()

    def generate(self, input_shape):
        '''
        generate the model
        '''
        self.model.add(Conv2D(32, kernel_size=(3, 3), strides=2, activation='relu', input_shape=input_shape))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(1000, activation='relu'))
        # 20 for number of classes
        self.model.add(Dense(20, activation='softmax'))
        
        return self.model




if __name__ == '__main__':
    
    rnn = RNN()
    model = rnn.generate(100, 50, 3, 20, 10)
    print(model)