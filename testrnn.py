from keras.models import Sequential
from keras import optimizers
model = Sequential()
model.add(LSTM(50, return_sequences=True))
model.add(Activation('softmax'))
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
