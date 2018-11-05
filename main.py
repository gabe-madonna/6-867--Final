from model import RNN
from preprocess import gen_letter_dict, partition, to_matrices
from utils import *
import argparse


def main():
    # go_home()
    letters = gen_letter_dict(norm_n=25)
    letters_train, letters_test = partition(letters, ratio=.2)
    X_train, y_train = to_matrices(letters_train)
    X_test, y_test = to_matrices(letters_test)
    # print(X_test[0])
    model = RNN()
    model.generate(hidden_size=50, input_shape=X_test[0].shape, output_dim=len(NUM2LET), layers=2)
    model.train(X_train, y_train, epochs=20)
    model.test(X_test, y_test)


if __name__ == '__main__':
    main()