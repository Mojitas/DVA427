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
        print(i)


if __name__ == '__main__':

    """
    versicolor1: 
    (x1 == short or long) and (x2 == medium or long) and (x3 == medium or long) and (x4 == short)
    
    versiolor2:
    (x1 == medium) and (x2 == short or medium) and (x3 == short) and (x4 == long)
    
    setosa:
    (x3 == short or medium) and (x4 == short) 
    
    virginica:
    (X2 == short or medium) and (x3 == long) (x4 == long)
    """

    for i in range(4):
        print(short(data_array[0, i]))
        print(medium(data_array[0, i]))
        print(long(data_array[0, i]))
        print("\n")
