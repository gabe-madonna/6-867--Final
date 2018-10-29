from model import gen_model, train_model, test_model
from preprocess import gen_letter_dict, partition, go_home
from utils import *


def main():
    go_home()
    letters = gen_letter_dict()
    letters_train, letters_test = partition(letters, r=.2)
    X_train, y_train = to_matrices(letters_train)
    X_test, y_test = to_matrices(letters_train)
    model = gen_model()
    train_model(model, X_train, y_train)
    test_model(model, X_test, y_test, plot=True)


if __name__ == '__main__':
    main()