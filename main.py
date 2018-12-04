from model import RNN, CNN
from preprocess import gen_letter_dict, partition, to_matrices
from utils import *


def main():
    # go_home()
    letters = gen_letter_dict(norm_n=25, all_letters=False)
    letters_train, letters_test = partition(letters, ratio=.2)
    X_train, y_train = to_matrices(letters_train)
    X_test, y_test = to_matrices(letters_test)
    X_train = X_train[:, :, :-1]
    X_test = X_test[:, :, :-1]

    # # test rnn
    # test_rnn(X_train, y_train, X_test, y_test)

    # test cnn
    test_cnn(X_train, y_train, X_test, y_test)

def test_rnn(X_train, y_train, X_test, y_test):
    # test RNN
    model = RNN()
    model.generate(hidden_size=25, input_shape=X_test[0].shape, output_dim=len(NUM2LET), layers=1)
    model.train(X_train, y_train, epochs=10)
    model.test(X_test, y_test)

def test_cnn(X_train, y_train, X_test, y_test):
    # test CNN
    model = CNN()
    model.generate(input_shape=(X_test[0].shape[0], X_test[0].shape[1], 1))
    model.train(X_train, y_train, epochs=10)
    model.test(X_test, y_test)



if __name__ == '__main__':
    main()