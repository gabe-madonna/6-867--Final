import numpy as np
import string
from utils import *

NUM2LET = {i+1: string.ascii_letters[i] for i in range(20)}


def gen_letter(fname):
    '''
    generate np array from fname
    :param fname (str): name of file
    :return letter (np.array): n by 3 matrix for letter
    '''
    letter = np.genfromtxt(fname, delimiter=",")
    letter = norm(letter, n)
    return letter


def norm(letter, n):
    raise NotImplementedError()
    return letter

def extract_f_num(fname):
    '''
    get the index of the letter from its file name
    :param fname (str): name of file
    :return num (int): index of file
    '''
    assert len(fname) == 15
    num = int(fname[7:11])
    return num


def gen_labels_dict(fname):
    '''
    generate a dict mapping letter index (one-indexed) to letter label
    :param fname:
    :return ind2let (dict):
    '''
    ind2let = {}
    keys = np.genfromtxt(fname, delimiter=",")
    for ind, key in enumerate(keys):
        ind2let[ind+1] = NUM2LET[int(key)]
    return ind2let


def gen_letter_dict():
    '''
    iterates over data in /data and generates a dict of the result
    :return letters (dict): maps letter to list of np arrays of that letter
    '''
    assert_home()
    data_dir = 'data'
    chdir(data_dir)

    ind2label = gen_labels_dict('labels.csv')
    letters = {letter: [] for letter in ind2label.values()}
    f_names = sorted([fname for fname in os.listdir() if fname[:6] == 'letter'])
    for fname in f_names:
        letter = gen_letter(fname)
        num = extract_f_num(fname)
        label = ind2label[num]
        letters[label].append(letter)

    chdir(data_dir, reverse=True)
    return letters


def partition(letters, ratio=.2):
    '''
    partitions letters into two dicts by partitioning each letter by ratio
    :return train, test (dict, dict): partitions of letters
    '''
    test, train = {}, {}

    for label in letters:
        n = len(letters[label])
        r = int(n*ratio)
        mask = r*[0] + (n-r)*[1]
        np.random.shuffle(mask)
        train[label] = mask_list(letters[label], mask)
        test[label] = mask_list(letters[label], mask, inverse=True)

    return train, test


def to_matrices(letters):
    '''
    takes letters dict and turns into
    :param letters: dict mapping letters to lists of arrays for those letters
    :return X, y: np.3darray and np.1darray of stacked letter matrices and their corresponding labels
    '''
    X = []
    y = []
    labels = sorted(letters.keys())
    for label in labels:
        X += letters[label]
        y += [label]*len(letters[label])
    X, y = np.array(X), np.array(y)
    y = y_encode(y)
    return X, y


def y_encode(y):
    '''
    one-hot encode multi-class label data
    :param y: np.1darray of letter labels
    :return one_hot: np.2darray of 1-hot encoded label vector
    '''
    let2num = dict((val, key) for (key, val) in NUM2LET.items())
    one_hot = np.zeros((len(y), max(NUM2LET.keys())))
    for i, yi in enumerate(y):
        ind = let2num[yi]
        one_hot[i][ind-1] = 1
    return one_hot
