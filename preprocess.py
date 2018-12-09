import numpy as np
from utils import *
from scipy import interpolate, ndimage
from keras.utils import to_categorical
import string


def gen_letter(fname, dataset, norm_n=None, derivative=False, integral=False):
    '''
    generate np array from fname
    :param fname (str): name of file
    :return letter (np.array): n by 3 matrix for letter
    '''
    if dataset == 1:
        letter = np.genfromtxt(fname, delimiter=",")
    elif dataset == 2:
        strokes = []
        strokei = 0
        fname = fname[:-38] + '{}' + fname[-37:]
        while True:
            try:
                stroke = np.genfromtxt(fname.format(strokei), delimiter=",")
                strokes.append(stroke)
                strokei += 1
            except:
                break
        try:
            letter = np.concatenate(strokes)
        except:
            print('couldnt concatenate', fname)
            return None

    # if derivative:
    #     letter = np.diff(letter, axis=0)
    # if integral:
    #     letter = np.trapz(letter, axis=0)
    letter = norm(letter, norm_n)
    return letter


def norm(letter, n=25, plot=False):
    '''
    :param letter:
    :param n:
    :param interp_kind:
    :return:
    '''
    if n is None:
        return letter

    f = None
    # spline norm of x and y
    if len(letter.T) == 3:
        x, y, f = letter.T
    else:
        assert len(letter.T) == 2
        x, y = letter.T

    safe_inds = np.where(np.abs(np.diff(x)) + np.abs(np.diff(y)) > 0)
    x2 = np.r_[x[safe_inds], x[-1]]
    y2 = np.r_[y[safe_inds], y[-1]]
    x4, y4 = x2, y2
    # jump = np.sqrt(np.diff(x2) ** 2 + np.diff(y2) ** 2)
    # smooth_jump = ndimage.gaussian_filter1d(jump, 5, mode='wrap')  # window of size 5 is arbitrary
    # limit = 2 * np.median(smooth_jump)  # factor 2 is arbitrary
    # x3, y3 = x2[:-1], y2[:-1]
    # x4 = x3[(jump > 0) & (smooth_jump < limit)]
    # y4 = y3[(jump > 0) & (smooth_jump < limit)]
    if len(x4) <= n:
        return None
        # mult = int(np.ceil(n/len(x4)) + 1)
        # x4, y4 = np.concatenate([x4 for i in range(mult)]), np.concatenate([y4 for i in range(mult)])
    tck, u = interpolate.splprep(np.array([x4, y4]), s=.001)
    unew = np.linspace(0, 1, n)
    norm_x, norm_y = interpolate.splev(unew, tck)

    if f is not None:
        # linear interpolation of force
        index = np.linspace(1, len(letter), len(letter))
        new_index = np.linspace(1, len(letter), n)
        # function f to interpolate column-wise and build normalized rep
        interp = interpolate.interp1d(index, f, kind='linear', assume_sorted=True)
        norm_f = interp(new_index)

    if plot:
        plt.figure()
        plt.plot(x, y, '-k', norm_x, norm_y)
        plt.legend(['True', 'Interpolation'])
        plt.title('Spline of parametrically-defined curve')
        plt.show()

        if f is not None:
            plt.figure()
            plt.plot(index, f)
            plt.plot(new_index, norm_f)
            plt.legend(['True', 'Interpolation'])
            plt.title('Interpolation of 1d Force Function')
            plt.show()

    if f is not None:
        norm_letter = np.array([norm_x, norm_y, norm_f]).T
    else:
        norm_letter = np.array([norm_x, norm_y]).T
    return norm_letter


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


def gen_letter_dict(dataset, norm_n=None, all_letters=True, deriv=False, integ=False, filterr=None):
    '''
    iterates over data in /data and generates a dict of the result
    :return letters (dict): maps letter to list of np arrays of that letter
    '''
    # assert_home()
    if dataset == 1:
        data_dir = 'data'
        os.chdir(data_dir)
        print("Generating labels")
        ind2label = gen_labels_dict('labels.csv')
        letters = {letter: [] for letter in ind2label.values()}
        f_names = sorted([fname for fname in os.listdir() if fname[:6] == 'letter'])
        print("Reading in letters")
        # limit the number of letters read in to speed things up
        if not all_letters:
            np.random.shuffle(f_names)
            f_names = f_names[:20]
        # read in each letter
        for fname in f_names:
            print('   Reading', fname)
            letter = gen_letter(fname, dataset, norm_n, deriv, integ)
            num = extract_f_num(fname)
            label = ind2label[num]
            letters[label].append(letter)
        os.chdir('..')
        print("Done reading letters")
    elif dataset == 2:
        letters = {}
        data_dir = 'data2'
        os.chdir(data_dir)
        f_names = set([fname for fname in os.listdir()])  # get first strokes
        print("Reading in letters")

        for fnamei, fname in enumerate(f_names):
            if fnamei % 100 == 0:
                if fnamei > 0 and not all_letters:
                    break
                print('   Reading character ' + str(fnamei))
            # print('   Reading', fname)
            label = fname.split('_')[1]
            if filterr is not None and label not in filterr:
                continue
            letter = gen_letter(fname, dataset, norm_n, deriv, integ)
            if letter is not None:
                letters.setdefault(label, []).append(letter)
            else:
                print('   Couldnt read', fname[:-37])
        os.chdir('..')
        print("Done reading letters")
    else:
        raise ValueError('dataset passed: ' + str(dataset))

    labels = sorted(letters.keys())
    y_map = {labels[i]: i for i in range(len(labels))}
    return letters, y_map


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


def to_matrices(letters, y_map):
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
    y = y_encode(y, y_map)
    return X, y


def y_encode(y, y_map):
    y_inds = np.array([y_map[yi] for yi in y])
    y_final = to_categorical(y_inds, len(y_map))
    return y_final



# def y_encode(y):
#     '''
#     one-hot encode multi-class label data
#     :param y: np.1darray of letter labels
#     :return one_hot: np.2darray of 1-hot encoded label vector
#     '''
#     let2num = dict((val, key) for (key, val) in NUM2LET.items())
#     one_hot = np.zeros((len(y), max(NUM2LET.keys())))
#     for i, yi in enumerate(y):
#         ind = let2num[yi]
#         one_hot[i][ind-1] = 1
#     return one_hot

