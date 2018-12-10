from model import RNN, CNN, KNN
from preprocess import gen_letter_dict, partition, to_matrices, y_encode
from utils import *
from sklearn import metrics
import numpy as np
import string


def main():
    filt = set(string.ascii_letters)
    # filt = {'a', 'A', 'F', 'f', 'C', 'c'}
    letters, y_map = gen_letter_dict(dataset=2, norm_n=15, all_letters=True,
            deriv=False, integ=False, filterr=filt)
    n_labels = len(y_map)
    letters_train, letters_test = partition(letters, ratio=.2)
    X_train, y_train = to_matrices(letters_train, y_map)
    X_test, y_test = to_matrices(letters_test, y_map)

    test_rnn(X_train, y_train, X_test, y_test, 250, n_labels, y_map)
    # test_cnn(X_train, y_train, X_test, y_test, 50)
    # if False:
    #     knnX_train = []
    #     knnX_test = []
    #     for elem in X_train:
    #         newArr = [point[1] for point in elem]
    #         newArr.extend([point[0] for point in elem])
    #         knnX_train.append(newArr)
    #     for elem in X_test:
    #         newArr = [point[1] for point in elem]
    #         newArr.extend([point[0] for point in elem])
    #         knnX_test.append(newArr)
    #     knn = KNN(8)
    #     knn.train(knnX_train, y_train)
    #     y_pred = knn.modelPredict(knnX_test)
    #     print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))


def test_rnn(X_train, y_train, X_test, y_test, epochs, n_labels, y_map):
    # test RNN
    NUM2LET = {value: key for (key, value) in y_map.items()}
    model = RNN()
    model.generate(NUM2LET=NUM2LET, hidden_size=25, input_shape=X_test[0].shape, output_dim=n_labels, layers=1)
    model.train(X_train, y_train, epochs=epochs)
    model.test(X_test, y_test)


def test_cnn(X_train, y_train, X_test, y_test, epochs, n_labels, y_map):
    # test CNN
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

    model = CNN()
    model.generate(input_shape=X_test[0].shape)
    model.train(X_train, y_train, epochs=epochs)
    acc = model.test(X_test, y_test)
    return acc


def test_template(X_train, y_train, X_test, y_test, epochs, n_labels, y_map):
    NUM2LET = {value: key for (key, value) in y_map.items()}
    model = RNN()
    model.generate(NUM2LET=NUM2LET, hidden_size=25, input_shape=X_test[0].shape, output_dim=n_labels, layers=2)
    model.train(X_train, y_train, epochs=epochs)
    model.test(X_test, y_test)


def debug():
    print('debugging')
    letters, y_map = gen_letter_dict(dataset=1, norm_n=25, all_letters=True, deriv=False, integ=False, filterr=set(string.ascii_letters))

    for char in ['a', 'b', 'c', 'w', 'm', 'p']:
        plot_letters(letters[char][0], label="{}".format(char))


if __name__ == '__main__':
    main()
    # debug()
    # pass

# debug()
