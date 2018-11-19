from model import RNN
from preprocess import gen_letter_dict, partition, to_matrices
from utils import *


def main():
    # go_home()
    letters = gen_letter_dict(norm_n=25)
    letters_train, letters_test = partition(letters, ratio=.2)
    X_train, y_train = to_matrices(letters_train)
    X_test, y_test = to_matrices(letters_test)
    X_train = X_train[:, :, :-1]
    X_test = X_test[:, :, :-1]
    model = RNN()
    model.generate(hidden_size=25, input_shape=X_test[0].shape, output_dim=len(NUM2LET), layers=1)
    model.train(X_train, y_train, epochs=60)
    model.test(X_test, y_test)


if __name__ == '__main__':
    main()