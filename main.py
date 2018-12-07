from model import RNN
from preprocess import gen_letter_dict, partition, to_matrices
from utils import *


def main():
    # go_home()
    letters = gen_letter_dict(norm_n=25, all_letters=False, deriv=False, integ=False)
    letters_train, letters_test = partition(letters, ratio=.2)
    X_train, y_train = to_matrices(letters_train)
    X_test, y_test = to_matrices(letters_test)
    X_train = X_train[:, :, :-1]
    X_test = X_test[:, :, :-1]
    model = RNN()
    model.generate(hidden_size=10, input_shape=X_test[0].shape, output_dim=len(NUM2LET), layers=3)
    model.train(X_train, y_train, epochs=100)
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