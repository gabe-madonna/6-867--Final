import keras


class KerasLSTM:

    def generate(self, hidden_size, timesteps, features, output_dim):
        model = Sequential()
        # 50 for number of timesteps, 3 for features
        model.add(LSTM(hidden_size, input_shape=(timesteps, features)))
        # dense takes in output dimensionality
        model.add(Dense(output_dim))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='sgd')

        return model