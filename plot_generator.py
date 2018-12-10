from model import RNN, CNN, KNN
from preprocess import gen_letter_dict, partition, to_matrices, y_encode
from utils import *
from sklearn import metrics
import numpy as np
import string
import time


def save_params(params):
    fname = 'plot_results.txt'
    with open(fname, 'a+') as f:
        f.write(str(params))


def run(params):
    try:

        if params['norm_n'] in datasets_dict[params['dataset']]:
            letters, y_map = datasets_dict[params['dataset']][params['norm_n']]
        else:
            (letters, y_map) = gen_letter_dict(dataset=params['dataset'], norm_n=params['norm_n'],
                                all_letters=True, filterr=set(string.ascii_letters))
            datasets_dict[params['dataset']][params['norm_n']] = (letters, y_map)

        NUM2LET = reverse_dict(y_map)
        n_labels = len(y_map)
        letters_train, letters_test = partition(letters, ratio=.2)
        X_train, y_train = to_matrices(letters_train, y_map)
        X_test, y_test = to_matrices(letters_test, y_map)

        if params['model'] in ('RNN', 'CNN'):
            if params['model'] == 'RNNN':
                model = RNN()
            else:
                model = CNN()

            model.generate(NUM2LET=NUM2LET, hidden_size=params['n_units'],
                           input_shape=X_test[0].shape, output_dim=n_labels,
                           layers=params['n_layers'])
            t = time.time()
            params['losses'], params['accuracies'], model.train(X_train, y_train, epochs=params['n_epochs'])
            params['runtime'] = time.time() - t
            params['loss'], params['accuracy'], params['error_dict'] = model.test(X_test, y_test)

        elif params['model'] == 'KNN':
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
            t = time.time()
            knn = KNN(params['knn_n'])
            knn.train(knnX_train, y_train)
            y_pred = knn.modelPredict(knnX_test)
            params['runtime'] = time.time() - t
            params['loss'], params['accuracy'], params['error_dict'] = osfdspafdj
            # print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
        elif params['model'] == 'Template Matching':
            do model stuff
        else:
            raise ValueError('Unrecognized model type: {}'.format(params['model']))

        save_params(params)
    except:
        print('couldnt execute for params \n{}'.format(str(params)))


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


