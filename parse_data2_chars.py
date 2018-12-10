import numpy as np
from utils import *
import uuid


def get_letter(lines, ind):
    while True:
        if lines[ind].strip()[1] == '/':
            ind += 1
        else:
            break
    if not lines[ind][:4] == 'WORD':
        raise ValueError(str(ind) + ': unexpected line: ' + lines[ind])
    label = lines[ind].strip().split(' ')[1]
    if not len(label) == 1:
        raise ValueError(str(ind) + ': label not length 1: ' + str(label))
    ind += 1
    n = int(lines[ind].strip().split(' ')[1])
    ind += 1
    strokes = []
    for i in range(n):
        assert lines[ind].strip().split(' ')[2] == '#'
        n_points = int(lines[ind].strip().split(' ')[1])
        points = lines[ind].strip().split(' ')[3:]
        xy = np.array([[int(points[j]), -1*int(points[j+1])] for j in range(0, len(points), 2)])
        if not len(xy) == n_points:
            raise ValueError(str(ind) + ': len(xy): ' + str(len(xy)) + ', expected:' + str(n_points))
        strokes.append(xy)
        ind += 1
    return label, strokes, ind


def extract_letters():
    in_name = 'raw_chars2.txt'
    out_name = 'data3\\{name}_{label}_{char_num:04d}_{stroke_num}_{hash_id}.txt'  # 'Stock_A_0043_stroke0'

    with open(in_name, encoding='utf8') as f:
        lines = f.readlines()

    print('Reading Letters')

    ind = 0
    letteri = 0

    letters_count = {}
    while ind < len(lines):
        letteri += 1
        hash_id = uuid.uuid4().hex
        label, strokes, ind = get_letter(lines, ind)
        if letteri % 100 == 0:
            print('Reading {} ({}) '.format(label, letteri))
        # letter_dict.setdefault(label, []).append(strokes)
        if label not in letters_count:
            letters_count[label] = 0
        char_num = letters_count[label]
        for strokei, stroke in enumerate(strokes):
            try:
                fname = out_name.format(name='Stock', label=label,
                                        char_num=char_num, stroke_num=strokei, hash_id=hash_id)
                np.savetxt(fname, stroke, delimiter=",")
            except:
                print('couldnt save letter {}: {}'.format(letteri, label))

        letters_count[label] += 1

    print('Done Reading')
    # return letter_dict


# def write_letters(letter_dict):
#     print('Writing Letters')
#     fname = 'Stock_{0}_{1:04d}_{2}.txt'  # 'Stock_A_0043_stroke0'
#     for char_n, (label, letters) in enumerate(letter_dict.items()):
#         print('Writing Letter {} [#{}]'.format(label, char_n+1))
#         for letteri, strokes in enumerate(letters):
#             time.sleep(.01)
#             for strokei, stroke in enumerate(strokes):
#                 try:
#                     np.savetxt(fname.format(label, letteri, strokei), stroke, delimiter=",")
#                 except:
#                     char_name = 'char{0:04d}'.format(char_n)
#                     np.savetxt(fname.format(char_name, letteri, strokei), stroke, delimiter=",")
#     print('Done Writing')


def normalize_means():
    fnames = set([fname[:-38]+'{}'+fname[-37:] for fname in os.listdir()])  # get first strokes
    for i, fname in enumerate(fnames):
        if i % 1000 == 0:
            print('reading file {}'.format(i))
        strokes = []
        strokei = 0
        while True:
            try:
                stroke = np.genfromtxt(fname.format(strokei), delimiter=",")
                strokes.append(stroke)
                strokei += 1
            except:
                break
        try:
            letter = np.concatenate(strokes)
            x_bar = np.mean(letter[:, 0])
            y_bar = np.mean(letter[:, 1])
            for strokei, stroke in enumerate(strokes):
                stroke[:, 0] -= x_bar
                stroke[:, 1] -= y_bar
                np.savetxt(fname.format(strokei), stroke, delimiter=",")
        except:
            pass


def get_std_mean():
    x, y = [], []
    fnames = os.listdir()
    for i, fname in enumerate(fnames):
        if i % 1000 == 0:
            print('reading file {}'.format(i))
        stroke = np.genfromtxt(fname, delimiter=",")
        x.append(stroke[:, 0])
        y.append(stroke[:, 1])
    x, y = np.concatenate(x), np.concatenate(y)
    sigma = np.std(np.concatenate((x, y)))
    x_bar, y_bar = np.mean(x), np.mean(y)
    return sigma, x_bar, y_bar


def normalize_std(sigma):
    fnames = os.listdir()
    for i, fname in enumerate(fnames):
        if i % 1000 == 0:
            print('reading file {}'.format(i))
        stroke = np.genfromtxt(fname, delimiter=",")
        stroke /= sigma
        np.savetxt(fname, stroke, delimiter=",")


def normalize_letters():

    print('normalizing means')
    normalize_means()

    print('getting values')
    sigma, x_avg, y_avg = get_std_mean()

    print('normalizing std')
    normalize_std(sigma)

    print('checking values')
    sigma, x_avg, y_avg = get_std_mean()
    assert abs(sigma - 1) < .001
    assert abs(x_avg) < .001
    assert abs(y_avg) < .001
    print('done')


if __name__ == '__main__':
    # letter_dict = extract_letters()
    # write_letters(letter_dict)
    os.chdir('data2')
    normalize_letters()
    os.chdir('..')
    pass
