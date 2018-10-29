import keras


class KerasLSTM:

    def generate(self, hidden_size, timesteps, features, output_dim):
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
        model.fit(train_x, train_y, epochs=epochs, validation_data=(test_x, test_y))

    def test(test_X, test_Y):
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