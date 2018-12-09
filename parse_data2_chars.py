import numpy as np
import os


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
        # xy[:, 1] = -1 * xy[:, 1][::-1]
        if not len(xy) == n_points:
            raise ValueError(str(ind) + ': len(xy): ' + str(len(xy)) + ', expected:' + str(n_points))
        strokes.append(xy)
        ind += 1
    # if ind < len(lines):
    #     assert not lines[ind].strip().split(' ')[2] == '#'
    return label, strokes, ind


def read_letters():
    filename = 'data2\\raw_chars.txt'
    letter_dict = {}

    with open(filename, encoding='utf8') as f:
        lines = f.readlines()

    print('Reading Letters')

    ind = 0
    letteri = 0
    while ind < len(lines):
        print('Reading Letter ', letteri)
        letteri += 1
        label, strokes, ind = get_letter(lines, ind)
        letter_dict.setdefault(label, []).append(strokes)
    print('Done Reading')
    return letter_dict


def write_letters(letter_dict):
    print('Writing Letters')
    char_n = 0
    fname = 'Stock_{0}_{1:04d}_{2}.txt'  # 'Stock_A_0043_stroke0'
    items = letter_dict.items()
    for label, letters in items:
        print('Writing Letter [[}/{}]'.format(char_n, len(items)))
        char_n += 1
        for letteri, strokes in enumerate(letters):
            for strokei, stroke in enumerate(strokes):
                try:
                    char_name = label
                    np.savetxt(fname.format(char_name, letteri, strokei), stroke, delimiter=",")
                except:
                    char_name = 'char{0:04d}'.format(char_n)
                    np.savetxt(fname.format(char_name, letteri, strokei), stroke, delimiter=",")
    print('Done Writing')


if __name__ == '__main__':
    # letter_dict = read_letters()
    # os.chdir('data2')
    # write_letters(letter_dict)
    pass
