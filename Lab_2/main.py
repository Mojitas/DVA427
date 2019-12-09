from lib_lab_2 import *


# Math functions that returns how much it belongs to each classification

def short(x):
    if x<0.6:
        return -5 * x / 3 + 1
    else:
        return 0

def medium(x):
    if x < 0.6:
        return 5 * x / 3
    else:
        return -5 * x / 2 + 5 / 2


def long(x):
    if x>0.6:
        return 5 * x / 2 - 1.5
    else:
        return 0

def parametrize(x):

    for i in range(4):
        x[i]=


if __name__ == '__main__':
    print(data_array[0, 0:4])

    print(short(data_array[0, 0:4]))
    print(low_medium(data_array[0, 0:4]))
    print(high_medium(data_array[0, 0:4]))
    print(long(data_array[0, 0:4]))
