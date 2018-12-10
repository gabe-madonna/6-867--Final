from model import RNN, CNN, KNN
from preprocess import gen_letter_dict, partition, to_matrices, y_encode
from utils import *
from sklearn import metrics
import numpy as np
import string

def save_params(params):
    pass


def run(params):

    if params['norm_n'] in datasets_dict[params['dataset']]:
        dataset = datasets_dict[params['dataset']][params['norm_n']]
    else:
        data_set = gen_letter_dict(dataset=params['dataset'], norm_n=params['norm_n'],
                            all_letters=True, filterr=set(string.ascii_letters))
        datasets_dict[params['dataset']][params['norm_n']] = data_set

    







    filt = set(string.ascii_letters)
    letters, y_map = gen_letter_dict(dataset=2, norm_n=15, all_letters=True, filterr=filt)
    n_labels = len(y_map)
    letters_train, letters_test = partition(letters, ratio=.2)
    X_train, y_train = to_matrices(letters_train, y_map)
    X_test, y_test = to_matrices(letters_test, y_map)
    test_rnn(X_train, y_train, X_test, y_test, 125, n_labels, y_map)
    # test_cnn(X_train, y_train, X_test, y_test, 50)
    if False:
        knnX_train = []
        knnX_test = []
        for elem in X_train:
            newArr = [point[1] for point in elem]
            newArr.extend([point[0] for point in elem])
            knnX_train.append(newArr)
        for elem in X_test:
            newArr = [point[1] for point in elem]
            newArr.extend([point[0] for point in elem])
            knnX_test.append(newArr)
        knn = KNN(8)
        knn.train(knnX_train, y_train)
        y_pred = knn.modelPredict(knnX_test)
        print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))


def main():

    params = {'model': 'RNN',
              'data_set': 0,
              'n_epochs': 0,
              'n_layers': 0,
              'n_units': 0,
              'norm_n': 0,
              'knn_n': 0,
              'error_dict': dict(),
              'losses': [],
              'accuracies': [],
              'loss': 0,
              'accuracy': 0,
              'runtime': 0
              }

    global datasets_dict
    datasets_dict = {1: {}, 2: {}}  # dataset[1][20] is letter dict for dataset 1, norm_n = 20

    models = ('RNN', 'CNN', 'KNN', 'Template Matching')
    datasets = (1, 2)
    n_epochs = (10, 25, 50, 100, 200)
    n_layers = (1, 2, 3, 4, 5)
    n_units = (5, 10, 25, 50, 75)
    norm_ns = (5, 10, 15, 20, 30)
    knn_ns = (1, 2, 4, 8, 16, 32)

    for model in models:
        params['model'] = model
        for dataset in datasets:
            if model in ('RNN', 'CNN'):

                params['n_layers'] = 1
                params['n_epochs'] = 100
                params['n_units'] = 25
                params['norm_n'] = 15

                for n_layer in n_layers:
                    params['n_layers'] = n_layer
                    run(params)

                params['n_layers'] = 1
                params['n_epochs'] = 100
                params['n_units'] = 25
                params['norm_n'] = 15

                for n_epoch in n_epochs:
                    params['n_epochs'] = n_epoch
                    run(params)

                params['n_layers'] = 1
                params['n_epochs'] = 100
                params['n_units'] = 25
                params['norm_n'] = 15

                for n_unit in n_units:
                    params['n_units'] = n_unit
                    run(params)

                params['n_layers'] = 1
                params['n_epochs'] = 100
                params['n_units'] = 25
                params['norm_n'] = 15

                for norm_n in norm_ns:
                    params['norm_n'] = norm_n
                    run(params)

            elif model == 'KNN':

                params['norm_n'] = 15
                params['knn_n'] = 8

                for knn_n in knn_ns:
                    params['knn_n'] = knn_n
                    run(params)

                params['norm_n'] = 15
                params['knn_n'] = 8

                for norm_n in norm_ns:
                    params['norm_n'] = norm_n
                    run(params)

            elif model == 'Template Matching':

                params['norm_n'] = 15

                for norm_n in norm_ns:
                    run(params)

            else:
                raise ValueError('Unrecognized model passed: {}'.format(model))








        try:
            plot(params)
        except:
            print('Couldnt plot', params)
