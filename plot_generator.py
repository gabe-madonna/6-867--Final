from utils import *
import numpy as np
import string
import time
import uuid



def save_params(params):
    print(params)
    fname = 'plot_results.txt'
    with open(fname, 'a+') as f:
        f.write(str(params) + '\n')

def filter_params(params, filters):
    good = []
    for p in params:
        add = True
        for key in filters:
            if type(filters[key]) in (list, tuple, set):
                if p[key] in filters[key]:
                    pass
                else:
                    add = False
                    break
            elif p[key] == filters[key]:
                pass
            else:
                add = False
                break
        if add:
            good.append(p)
    return good


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


def main_data():

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

                params['n_layers'] = 2
                params['n_epochs'] = 20
                params['n_units'] = 25
                params['norm_n'] = 15

                for n_layer in n_layers[-2:]:
                    params['n_layers'] = n_layer
                    run(params)

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


def main_plot():
    params = []
    with open('plot_results.txt') as f:
        for line in f.readlines():
            print(line)
            params.append(eval(line))



    # plot letters
    # letters, y_map = gen_letter_dict(dataset=2, norm_n=15, all_letters=True, filterr=set(string.ascii_letters))
    # for letter in letters:
    #     # time.sleep(1)
    #     plot_letters(letters[letter], label='{}'.format(letter), alpha=.2, plot_avg=True)
    #



    # bargraph error by model, dataset

    # accuracies1 = dict()
    # for model in ('RNN', 'CNN', 'KNN', 'Template Matching'):
    #     print(model)
    #     accuracies1[model] = max([p['accuracy'] for p in filter_params(params, {'model': model, 'data_set': 1})])
    #
    # accuracies2 = dict()
    # for model in ('RNN', 'CNN', 'KNN', 'Template Matching'):
    #     print(model)
    #     accuracies2[model] = max([p['accuracy'] for p in filter_params(params, {'model': model, 'data_set': 2})])
    #
    # to_plot = sorted(list(accuracies1.keys()), key=lambda k: accuracies1[k])
    # y_pos = arange(len(to_plot))
    # performance = [accuracies1[model] for model in to_plot]
    # plt.bar(y_pos, performance, align='center', alpha=1, color='seagreen')
    # plt.xticks(y_pos, to_plot)
    # plt.ylabel('Accuracy')
    # plt.xlabel('Model')
    # title = 'Model Accuracies (Dataset 1)'
    # plt.figtext(.5, .9, title, fontsize=18, ha='center')
    # plt.savefig('figures\\model_accuracy_on_data_set_{}_{}.png'.format(1, uuid.uuid4().hex))
    # plt.show()
    # plt.close('all')
    #
    # to_plot = sorted(list(accuracies1.keys()), key=lambda k: accuracies1[k])
    # y_pos = arange(len(to_plot))
    # performance = [accuracies2[model] for model in to_plot]
    # plt.bar(y_pos, performance, align='center', alpha=1, color='seagreen')
    # plt.xticks(y_pos, to_plot)
    # plt.ylabel('Accuracy')
    # plt.xlabel('Model')
    # title = 'Model Accuracies (Dataset 2)'
    # plt.figtext(.5, .9, title, fontsize=18, ha='center')
    # plt.savefig('figures\\model_accuracy_on_data_set_{}_{}.png'.format(2, uuid.uuid4().hex))
    # plt.show()
    # plt.close('all')



    # graph accuracy by epochs in RNN dataset 2
    # accuracies = []
    # epochs = (10, 20, 40, 70, 100)
    # for epoch in epochs:
    #     print(epoch)
    #     accuracies += [max([p['accuracy'] for p in filter_params(params,
    #             {'model': 'RNN', 'norm_n': 15, 'n_layers': 1, 'data_set': 2, 'n_epochs': epoch})])]
    #
    # plt.plot(epochs, accuracies)
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epochs')
    # title = 'RNN Accuracies by Epoch'
    # subtitle = '(Dataset 2, 15 segments, 1 layer)'
    # plt.figtext(.5, .95, title, fontsize=18, ha='center')
    # plt.figtext(.5, .9, subtitle, fontsize=10, ha='center')
    # plt.savefig('figures\\accuracy_by_epochs_rnn_{}.png'.format(uuid.uuid4().hex))
    # plt.show()
    # plt.close('all')



    # graph accuracy by number of segments

    # accuracies_rnn = []
    # accuracies_cnn = []
    # accuracies_knn = []
    # accuracies_template = []
    #
    # norm_ns = (5, 10, 15, 20, 30)
    # for norm_n in norm_ns:
    #     print(norm_n)
    #     accuracies_rnn += [max([p['accuracy'] for p in filter_params(params, {'model': 'RNN', 'norm_n': norm_n, 'data_set': 2})])]
    #     accuracies_cnn += [max([p['accuracy'] for p in filter_params(params, {'model': 'CNN', 'norm_n': norm_n,'data_set': 2})])]
    #     accuracies_knn += [max([p['accuracy'] for p in filter_params(params, {'model': 'KNN', 'norm_n': norm_n, 'data_set': 2})])]
    #     accuracies_template += [max([p['accuracy'] for p in filter_params(params, {'model': 'Template Matching', 'norm_n': norm_n, 'data_set': 2})])]
    #
    # plt.plot(norm_ns, accuracies_rnn, label='RNN')
    # plt.plot(norm_ns, accuracies_cnn, label='CNN')
    # plt.plot(norm_ns, accuracies_knn, label='KNN')
    # plt.plot(norm_ns, accuracies_template, label='Template Matching')
    #
    # plt.ylabel('Accuracy')
    # plt.xlabel('N Segments')
    # title = 'Accuracies by Number of Letter Segments'
    # subtitle = '(Dataset 2)'
    # plt.figtext(.5, .95, title, fontsize=18, ha='center')
    # plt.figtext(.5, .9, subtitle, fontsize=10, ha='center')
    # plt.legend()
    # plt.savefig('figures\\accuracy_by_norm_n_dataset_2_{}.png'.format(uuid.uuid4().hex))
    # plt.show()
    # plt.close('all')
    # accuracies_rnn = []
    # accuracies_cnn = []
    # accuracies_knn = []
    # accuracies_template = []
    #
    # norm_ns = (5, 10, 15, 20, 30)
    # for norm_n in norm_ns:
    #     print(norm_n)
    #     accuracies_rnn += [
    #         max([p['accuracy'] for p in filter_params(params, {'model': 'RNN', 'norm_n': norm_n, 'data_set': 1})])]
    #     accuracies_cnn += [
    #         max([p['accuracy'] for p in filter_params(params, {'model': 'CNN', 'norm_n': norm_n, 'data_set': 1})])]
    #     accuracies_knn += [
    #         max([p['accuracy'] for p in filter_params(params, {'model': 'KNN', 'norm_n': norm_n, 'data_set': 1})])]
    #     accuracies_template += [max([p['accuracy'] for p in filter_params(params, {'model': 'Template Matching', 'norm_n': norm_n, 'data_set': 1})])]
    #
    # plt.plot(norm_ns, accuracies_rnn, label='RNN')
    # plt.plot(norm_ns, accuracies_cnn, label='CNN')
    # plt.plot(norm_ns, accuracies_knn, label='KNN')
    # plt.plot(norm_ns, accuracies_template, label='Template Matching')
    #
    # plt.ylabel('Accuracy')
    # plt.xlabel('N Segments')
    # title = 'Accuracies by Number of Letter Segments'
    # subtitle = '(Dataset 1)'
    # plt.figtext(.5, .95, title, fontsize=18, ha='center')
    # plt.figtext(.5, .9, subtitle, fontsize=10, ha='center')
    # plt.legend()
    # plt.savefig('figures\\accuracy_by_norm_n_dataset_1_{}.png'.format(uuid.uuid4().hex))
    # plt.show()
    # plt.close('all')



    # graph accuracy by number of layers for rnn and cnn

    # accuracies_rnn = []
    # accuracies_cnn = []
    #
    # layers = (1, 2, 3, 4)
    # for layer in layers:
    #     print(layer)
    #     accuracies_rnn += [max([p['accuracy'] for p in filter_params(params, {'model': 'RNN', 'n_layers': layer, 'norm_n': 15, 'data_set': 1})])]
    #     accuracies_cnn += [max([p['accuracy'] for p in filter_params(params, {'model': 'CNN', 'n_layers': layer, 'norm_n': 15, 'data_set': 1})])]
    #
    # plt.plot(layers, accuracies_rnn, label='RNN')
    # plt.plot(layers, accuracies_cnn, label='CNN')
    #
    # plt.ylabel('Accuracy')
    # plt.xlabel('Number of Layers')
    # title = 'Accuracies by Number of Model Layers'
    # subtitle = '(Dataset 1, 20 Epochs, 15 Segments)'
    # plt.figtext(.5, .95, title, fontsize=18, ha='center')
    # plt.figtext(.5, .9, subtitle, fontsize=10, ha='center')
    # plt.legend()
    # plt.savefig('figures\\accuracy_by_layers_dataset_1_{}.png'.format(uuid.uuid4().hex))
    # plt.show()
    # plt.close('all')



    # graph error by letter
    dicts = filter_params(params, {'model': 'RNN', 'data_set': 2})
    error_dict_rnn = sorted(dicts, key=lambda p: p['accuracy'])[-1]['error_dict']  # get highest accuracy one

    dicts = filter_params(params, {'model': 'CNN', 'data_set': 2})
    error_dict_cnn = sorted(dicts, key=lambda p: p['accuracy'])[-1]['error_dict']  # get highest accuracy one

    dicts = filter_params(params, {'model': 'KNN', 'data_set': 2})
    error_dict_knn = sorted(dicts, key=lambda p: p['accuracy'])[-1]['error_dict'][1]  # get highest accuracy one

    dicts = filter_params(params, {'model': 'Template Matching', 'data_set': 2})
    error_dict_template = sorted(dicts, key=lambda p: p['accuracy'])[-1]['error_dict']  # get highest accuracy one

    to_plot = sorted(list(string.ascii_letters))
    x_pos = arange(len(to_plot))

    performance_rnn = [error_dict_rnn.get(letter, 0) for letter in to_plot]
    performance_cnn = [error_dict_cnn.get(letter, 0) for letter in to_plot]
    performance_knn = [error_dict_knn.get(letter, 0) for letter in to_plot]
    performance_template = [error_dict_template.get(letter, 0) for letter in to_plot]

    ax = plt.subplot(111)
    # ax.bar(x_pos, performance_rnn, align='center', alpha=.5, label='RNN')
    # ax.bar(x_pos, performance_cnn, align='center', alpha=.5, label='CNN', bottom=performance_rnn)
    # ax.bar(x_pos, performance_knn, align='center', alpha=.5, label='KNN',
    #        bottom=np.sum((performance_rnn, performance_cnn), axis=0))
    # ax.bar(x_pos, performance_template, align='center', alpha=.5, label='Template',
    #        bottom=np.sum((performance_rnn, performance_cnn, performance_knn), axis=0))


    ax.bar(x_pos, performance_knn, align='center', alpha=.5, label='KNN')
    ax.bar(x_pos, performance_cnn, align='center', alpha=.5, label='CNN', bottom=performance_knn)
    ax.bar(x_pos, performance_rnn, align='center', alpha=.5, label='RNN',
           bottom=np.sum((performance_knn, performance_cnn), axis=0))

    plt.xticks(x_pos, to_plot)
    plt.ylabel('Error')
    plt.xlabel('Letter')
    title = 'Error by Letter'
    subtitle = '(Dataset 2)'
    plt.figtext(.5, .95, title, fontsize=18, ha='center')
    plt.figtext(.5, .9, subtitle, fontsize=10, ha='center')
    plt.legend()
    plt.savefig('figures\\accuracy_by_letters_dataset_2_{}.png'.format(uuid.uuid4().hex))
    plt.show()
    plt.close('all')


if __name__ == '__main__':
    main_plot()
    # main_data()


