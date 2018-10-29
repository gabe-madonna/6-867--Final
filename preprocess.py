import numpy as np
import os

HOME = '6-867--Final'
NUM2LET = {1: 'a', 2: 'b', 3: 'c'}

def gen_letter(fname):
    letter = np.genfromtxt(fname, delimiter=",")
    return letter


def extract_f_num(fname):
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


def assert_home():
    '''
    safe way to make sure that current working directory is home
    :return None
    '''
    curr = os.getcwd()
    if not split_dir(curr)[-1] == HOME:
        raise ValueError('\nCurrent Directory:\n' + curr + '\nExpected: ../' + HOME)


def split_dir(dirr):
    '''
    get list form of directory string componenets(without any empty strings)
    :param dirr:
    :return:
    '''
    dir_list = list(filter(lambda d: d, dirr.split('\\')))
    return dir_list


def go_home():
    '''
    safe way to go to home directory
    :return:
    '''
    curr_list = split_dir(os.getcwd())
    ind = curr_list.index(HOME)
    if ind == -1:
        raise ValueError('Cannot find', HOME, 'in current path')
    else:
        n_levels = len(curr_list) - ind - 1
        for i in range(n_levels):
            os.chdir('..')


def chdir(dirr, reverse=False):
    '''
    easy, safe way to go into and out of directories
    :param dirr: directory to go into or out of
    :param reverse: return from directory if True, otherwise go into it
    :return: None
    '''
    if reverse:
        dirr_list = split_dir(dirr)
        if '..'in dirr_list:
            raise ValueError('unexpected directory "\\..": unreversable')
        n_levels = len(dirr_list)
        cwd = os.getcwd()
        if not cwd.split('\\')[-n_levels:] == dirr_list:
            raise ValueError('Current Directory:\n'+cwd+'\nExpected Directory:\n'+dirr)
        dirr = '..\\' * n_levels
    os.chdir(dirr)


def mask_list(X_list, mask, inverse=False):
    '''
    Makes it easy to filter an iterable using a mask of ones and zeros
    :param X_list: iterable to be filtered
    :param mask: iterable of booleans (or values to be evaluated as booleans)
    :param inverse: boolean for whether to return the inverse of the mask
    :return masked_list: subset of X_list after masking
    '''
    if not len(mask) == len(X_list):
        raise AttributeError('Length mismatch - X_list:', str(len(X_list)), 'mask:', str(len(mask)))
    if inverse:
        mask = [not bool(mask[i]) for i in range(len(mask))]
    masked_list = [X_list[i] for i in range(len(mask)) if mask[i]]
    return masked_list
