import os

HOME = '6-867--Final'


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