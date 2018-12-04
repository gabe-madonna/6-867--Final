import os
import string
import matplotlib.pyplot as plt


HOME = '6-867--Final'
LETTERS = ['a',  'b', 'c',  'd',  'e',  'g',  'h', 'l',  'm',  'n',  'o',  'p',  'q',  'r',  's',  'u',  'v',  'w',  'y',  'z']
NUM2LET = {i+1: LETTERS[i] for i in range(len(LETTERS))}
LET2NUM = {val: key for key, val in NUM2LET.items()}



def plot_letter(letter, label=None):
    '''
    plot a given letter
    :param letter (np.2darray): letter array
    :param label (str): letter type
    :return None:
    '''
    x, y, f = letter.T
    plt.plot(x, y)
    if label is not None:
        plt.title(label)
    plt.show()


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
