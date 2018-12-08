from model import RNN, CNN
from preprocess import gen_letter_dict, partition, to_matrices
from utils import *
import numpy as np


def main():
    # go_home()
    letters = gen_letter_dict(norm_n=25, all_letters=False, deriv=False, integ=False)
    letters_train, letters_test = partition(letters, ratio=.2)
    X_train, y_train = to_matrices(letters_train)
    X_test, y_test = to_matrices(letters_test)
    X_train = X_train[:, :, :-1]
    X_test = X_test[:, :, :-1]

    # # test rnn
    test_rnn(X_train, y_train, X_test, y_test, 50)

    # test cnn
    # test_cnn(X_train, y_train, X_test, y_test, 50)


def test_rnn(X_train, y_train, X_test, y_test, epochs):
    # test RNN
    model = RNN()
    model.generate(hidden_size=25, input_shape=X_test[0].shape, output_dim=len(NUM2LET), layers=1)
    model.train(X_train, y_train, epochs=epochs)
    model.test(X_test, y_test)


def test_cnn(X_train, y_train, X_test, y_test, epochs):
    # test CNN
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

    model = CNN()
    model.generate(input_shape=X_test[0].shape)
    model.train(X_train, y_train, epochs=epochs)
    model.test(X_test, y_test)


def debug():
    print('debugging')
    letters = gen_letter_dict(norm_n=25, all_letters=False, deriv=False, integ=False)
    letter = list(filter(lambda x: len(x) > 0, list(letters.values())))[0][0]
    plot_letter(letter)


if __name__ == '__main__':
    # main()
    pass

debug()