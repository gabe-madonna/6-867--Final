import numpy as np
import string
from utils import *

NUM2LET = {i+1: string.ascii_letters[i] for i in range(20)}


def gen_letter(fname):
    '''
    generate np array from fname
    :param fname (str):
    :return letter (np.array):
    '''
    letter = np.genfromtxt(fname, delimiter=",")
    return letter


def extract_f_num(fname):
    '''
    get the index of the letter from its file name
    :param fname (str):
    :return num (int):
    '''
    assert len(fname) == 15
    num = int(fname[7:11])
    return num


def gen_labels_dict(fname):
    '''
    generate a dict mapping letter index (one-indexed) to letter label
    :param fname:
    :return:
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

    # build masks by exercise, then concatenate into aggregate mask
    for label in letters:
        n = len(letters[label])
        r = int(n*ratio)
        mask = r*[0] + (n-r)*[1]
        np.random.shuffle(mask)
        train[label] = mask_list(letters[label], mask)
        test[label] = mask_list(letters[label], mask, inverse=True)

    return train, test


def mask_list(letters, mask, inverse=False):
    '''
    Makes it easy to filter an iterable using a mask of ones and zeros
    :param letters (iterable): iterable to be filtered
    :param mask (iterable): iterable of booleans
    :param inverse (bool): whether to return the inverse of the mask
    :return masked_list (list): subset of letters after masking
    '''
    if not len(mask) == len(letters):
        raise AttributeError('Length mismatch - letters:', str(len(letters)), 'mask:', str(len(mask)))
    if inverse:
        mask = [not bool(mask[i]) for i in range(len(mask))]
    masked_list = [letters[i] for i in range(len(mask)) if mask[i]]
    return masked_list
