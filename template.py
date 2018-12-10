from preprocess import gen_letter_dict, partition, to_matrices, y_encode
from utils import *
from scipy.spatial.distance import cdist
import numpy as np
import string
import datetime

# get datapoints
# normalize each data point to 25 points
# find average for each data point
# find absolute distance avg(|A - B|) 


class Template():
    def __init__(self):
        self.letters = []
        self.averages = []
        self.averages_dict = []

    def average_letters(self, letters_train):
        '''
        generate average matrices for each letter
        :param letters (dict): dictionary of letter to list of matrices
        :return keys, avgs, averages: letters, averages, and average dict
        '''
        averages = {}
        keys, avgs = [], []

        for letter in letters_train:
            samples = letters_train[letter]
            avg = np.mean(samples, axis=0)
            averages[letter] = avg
            keys.append(letter)
            avgs.append(avg)

        self.letters = keys
        self.averages = avgs
        self.averages_dict = averages
        
        return keys, avgs, averages

    def find_closest(self, sample, distance_metric):
        '''
        find the index of the closest matrix using distance_metric
        :param sample (np.array): number of the dataset
        '''
        dist = []
        for avg in self.averages:
            dist.append(np.average(cdist(sample, avg, metric=distance_metric)))

        ind = np.argmin(dist)

        return ind, dist

    def test_letters(self, test_X, test_y, num2let, distance_metric):
        predicted = []
        incorrect = 0
        for index, x in enumerate(test_X):
            ind, dist = self.find_closest(x, distance_metric)
            correct_ind = np.argmax(y_test[index])
            predicted.append((self.letters[ind], num2let[correct_ind]))
            if self.letters[ind] == num2let[correct_ind]:
                incorrect +=1

        incorrect /= len(test_y)

        return predicted, incorrect

def generate_train_test(dataset_num):
    '''
    generate train and test data
    :param dataset_num: number of the dataset
    :return letters_train, letters_test: data from dataset partioned into train and test
    '''
    letters, y_map = gen_letter_dict(dataset=dataset_num, norm_n=25, all_letters=True, deriv=False, integ=False, filterr=set(string.ascii_letters))

    letters_train, letters_test = partition(letters, ratio=.2)

    return letters_train, letters_test, y_map

def write_to_file(losses, dataset):
    best = min(losses, key = lambda t: t[1])
    with open("results.txt", "a") as myfile:
        myfile.write("-------------------\n")
        myfile.write("TEMPLATE MATCHING: DATASET {}\n".format(dataset))
        myfile.write(str(datetime.datetime.now()) + '\n')
        for metric, loss in losses:
            myfile.write('metric: {}, loss: {}, acc: {}\n'.format(metric, loss, 1-loss))
        myfile.write('BEST: metric: {}, loss: {}, acc: {}\n'.format(best[0], best[1], 1-best[1]))


if __name__ == "__main__":
    dataset = 2
    letters_train, letters_test, y_map = generate_train_test(dataset)
    num2let = {value: key for (key, value) in y_map.items()}

    X_train, y_train = to_matrices(letters_train, y_map)
    X_test, y_test = to_matrices(letters_test, y_map)

    template = Template()

    letters, averages, average_d = template.average_letters(letters_train)
    sample = X_test[0]

    # ind, dist = template.find_closest(sample, averages, distance_metric='seuclidean')
    # correct_ind = np.argmax(y_test[0])

    metrics = ['euclidean', 'seuclidean', 'sqeuclidean', 'cosine', 'correlation', 'chebyshev', 'canberra', 'mahalanobis']

    losses = []
    for metric in metrics:
        predicted, loss = template.test_letters(X_test, y_test, num2let, distance_metric=metric)
        print(metric, loss)
        losses.append((metric, loss))

    write_to_file(losses, dataset)

    
