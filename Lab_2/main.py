from lib_lab_2 import *


# Math functions that returns how much each x_i belongs to each classification
def short(x):
    if x <= 0.6:
        return -(5 * x) / 3 + 1
    else:
        return 0


def medium(x):
    if x <= 0.6:
        return 5 * (x / 3)
    else:
        return -(5 * x) / 2 + (5 / 2)


def long(x):
    if x >= 0.6:
        return (5 * x) / 2 - 3 / 2
    else:
        return 0


def compute_flower(x):  # takes all four parameters of one flower
    x_temp = np.zeros((4, 1))
    x_temp[0] = min(max(short(x[0]), long(x[0])), max(medium(x[1]), long(x[1])), max(medium(x[2]), long(x[2])),
                    medium(x[3]))  # rule 1 Versicolor 1

    x_temp[1] = min(max(short(x[2]), medium(x[2])), short(x[3]))  # rule 2    Setosa

    x_temp[2] = min(max(short(x[1]), medium(x[1])), long(x[2]), long(x[3]))  # rule 3 Virginica

    x_temp[3] = min(medium(x[0]), max(short(x[1]), medium(x[1])), short(x[2]), long(x[3]))  # rule 4 Versicolor 2
    return x_temp


def classify(x):  # classifies one flower
    x_temp = compute_flower(x)  # computes stuff
    y_temp = max(x_temp)
    if y_temp == x_temp[0] or y_temp == x_temp[3]:  # Versicolor
        return 2

    elif y_temp == x_temp[1]:  # Setosa
        return 1

    elif y_temp == x_temp[2]:  # virginica
        return 3

    else:  # error
        print("Something happened in classify!")
        return -1


def compare(x):  # Takes the whole data set and compares it to the output
    accuracy = 0
    for i in range(x.shape[0]):
        if classify(x[i, 0:4]) == x[i, 4]:
            accuracy += 1

    return accuracy / x.shape[0]


if __name__ == '__main__':
    acc = compare(data_array)  ## changed lots of shit
    print("acc:", acc)
