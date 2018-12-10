from model import RNN, CNN, KNN, Template
from preprocess import gen_letter_dict, partition, to_matrices, y_encode
from utils import *
from sklearn import metrics
import numpy as np
import string
import time


def save_params(params):
    print(params)
    fname = 'plot_results.txt'
    with open(fname, 'a+') as f:
        f.write(str(params) + '\n')


def run(params):
    # try:

    if params['norm_n'] in datasets_dict[params['data_set']]:
        letters, y_map = datasets_dict[params['data_set']][params['norm_n']]
    else:
        (letters, y_map) = gen_letter_dict(dataset=params['data_set'], norm_n=params['norm_n'],
                            all_letters=True, filterr=set(string.ascii_letters))
        datasets_dict[params['data_set']][params['norm_n']] = (letters, y_map)

    NUM2LET = reverse_dict(y_map)
    n_labels = len(y_map)
    letters_train, letters_test = partition(letters, ratio=.2)
    X_train, y_train = to_matrices(letters_train, y_map)
    X_test, y_test = to_matrices(letters_test, y_map)

    if params['model'] in ('RNN', 'CNN'):
        if params['model'] == 'RNN':
            model = RNN()
        else:
            model = CNN()
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], -1, 1))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], -1, 1))

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
            knnX_train.append(np.concatenate((elem[:, 0], elem[:, 1])))
        for elem in X_test:
            knnX_test.append(np.concatenate((elem[:, 0], elem[:, 1])))
        t = time.time()
        knn = KNN(params['knn_n'], NUM2LET)
        knn.train(knnX_train, y_train)
        params['error_dict'] = knn.get_stats(knnX_test, y_test)
        params['runtime'] = time.time() - t

    elif params['model'] == 'Template Matching':
        t = time.time()
        template = Template()
        template.average_letters(letters_train)
        params['runtime'] = time.time() - t
        params['accuracy'], params['error_dict'] = template.test_letters(X_test, y_test, NUM2LET, params['distance_metric'])

    else:
        raise ValueError('Unrecognized model type: {}'.format(params['model']))

    save_params(params)
    # except:
    #     print('couldnt execute for params \n{}'.format(str(params)))


def main():

    params = {'model': 'RNN',
              'data_set': 0,
              'n_epochs': 0,
              'n_layers': 0,
              'n_units': 0,
              'layers': (0,),
              'norm_n': 0,
              'knn_n': 0,
              'error_dict': dict(),
              'losses': [],
              'accuracies': [],
              'loss': 0,
              'accuracy': 0,
              'runtime': 0,
              'distance_metric': ''
              }

    global datasets_dict
    datasets_dict = {1: {}, 2: {}}  # dataset[1][20] is letter dict for dataset 1, norm_n = 20

    models = ('RNN', 'CNN', 'KNN', 'Template Matching')
    datasets = (1, 2)
    n_epochs = (2, 5, 10, 20, 30)
    n_layers = (1, 2, 3, 4, 5)
    n_units = (5, 10, 25, 50, 75)
    norm_ns = (5, 10, 15, 20, 30)
    knn_ns = (1, 2, 4, 8, 16, 32, 64, 128)
    distance_metrics = ('euclidean', 'seuclidean', 'chebyshev', 'mahalanobis')

    for model in models[3:]:
        params['model'] = model
        for dataset in datasets:
            params['data_set'] = dataset
            if model in ('RNN', 'CNN'):

                # params['n_layers'] = 2
                # params['n_epochs'] = 20
                # params['n_units'] = 25
                # params['norm_n'] = 15
                #
                # for n_layer in n_layers[-2:]:
                #     params['n_layers'] = n_layer
                #     run(params)

                params['n_layers'] = 1
                params['n_epochs'] = 20
                params['n_units'] = 25
                params['norm_n'] = 15

                for n_epoch in n_epochs:
                    params['n_epochs'] = n_epoch
                    run(params)

                params['n_layers'] = 1
                params['n_epochs'] = 20
                params['n_units'] = 25
                params['norm_n'] = 15

                for n_unit in n_units:
                    params['n_units'] = n_unit
                    run(params)

                params['n_layers'] = 1
                params['n_epochs'] = 20
                params['n_units'] = 25
                params['norm_n'] = 15

                for norm_n in norm_ns:
                    params['norm_n'] = norm_n
                    run(params)

            elif model == 'KNN':

                params['norm_n'] = 15
                params['knn_n'] = 16

                for knn_n in knn_ns:
                    params['knn_n'] = knn_n
                    run(params)

                params['norm_n'] = 15
                params['knn_n'] = 16

                for norm_n in norm_ns:
                    params['norm_n'] = norm_n
                    run(params)

            elif model == 'Template Matching':

                params['norm_n'] = 15
                params['distance_metric'] = 'seuclidean'

                for norm_n in norm_ns:
                    params['norm_n'] = norm_n
                    run(params)

                params['norm_n'] = 15
                params['distance_metric'] = 'seuclidean'

                for distance_metric in distance_metrics:
                    params['distance_metric'] = distance_metric
                    run(params)

            else:
                raise ValueError('Unrecognized model passed: {}'.format(model))

if __name__ == '__main__':
    main()


