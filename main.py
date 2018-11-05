from model import RNN
from preprocess import gen_letter_dict, partition, to_matrices
from utils import *


def main():
    go_home()
    letters = gen_letter_dict(norm_n=25)
    letters_train, letters_test = partition(letters, ratio=.2)
    X_train, y_train = to_matrices(letters_train)
    X_test, y_test = to_matrices(letters_train)
    model = RNN()
    model.generate(hidden_size=50, timesteps=X_train.shape[1], features=X_train.shape[2], output_dim=len(NUM2LET), layers=1)
    model.train(X_train, y_train, epochs=10)
    model.test(X_test, y_test)


# if __name__ == '__main__':
#     main()