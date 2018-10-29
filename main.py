from model import KerasLSTM
from preprocess import gen_letter_dict, partition, to_matrices
from utils import *


def main():
    go_home()
    letters = gen_letter_dict()
    letters_train, letters_test = partition(letters, ratio=.2)
    X_train, y_train = to_matrices(letters_train)
    X_test, y_test = to_matrices(letters_train)
    model = KerasLSTM()
    model.generate()
    model.train(X_train, y_train)
    model.test(X_test, y_test, plot=True)


if __name__ == '__main__':
    main()