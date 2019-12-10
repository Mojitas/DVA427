import numpy as np
import random as rng
import pandas as pd
import math

amount = 30  # Something divisible by three

data_array = np.array(pd.read_csv("berlin52.tsp", header=None))
data_array = data_array.astype(int)


def init():
    salesmen = np.zeros((amount, 54))

    for i in range(52):
        salesmen[:, i] = i + 1

    salesmen[:, 52] = 1

    salesmen = salesmen.astype(int)

    for i in range(amount):
        rng.shuffle(salesmen[i, 0:52])

    salesmen[:, 52] = salesmen[:, 0]

    return salesmen


def calculate(salesmen):
    for j in range(amount):

        for i in range(52):
            x1 = data_array[salesmen[j, i - 1] - 1, 1]
            y1 = data_array[salesmen[j, i - 1] - 1, 2]

            x2 = data_array[salesmen[j, i] - 1, 1]
            y2 = data_array[salesmen[j, i] - 1, 2]

            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            salesmen[j, 53] += distance

    return salesmen


def elitism(salesmen):
    elite = np.zeros((20, 54))

    elite = elite.astype(int)

    for j in range(20):

        tempmin = min(salesmen[:, 53])

        for i in range(amount):

            if tempmin == salesmen[i, 53]:
                tempmini = i

        elite[j] = salesmen[tempmini]

        # print(elite)

        salesmen[tempmini, 53] = 100000

    return elite


def cross(parent1, parent2):
    child = np.zeros((1, 54))

    remaining = np.zeros((1, 54))

    for j in range(10):

        randomtemp = rng.randint(0, 49);

        for i in range(4):
            child[0, randomtemp + i] = parent1[0, randomtemp + i]

        empty = 0

    for i in range(53):

        if parent2[0, i] not in child:
            remaining[0, empty] = parent2[0, i]
            empty += 1

    j = 0

    for i in range(53):

        if child[0, i] == 0:

            child[0, i] = remaining[0, j]
            j += 1;

    child[0, 52] = child[0, 0]

    return child


def crossover(elite):
    np.random.shuffle(elite)

    children = np.zeros((10, 54))

    for i in range(10):
        children[i] = cross(elite[i:i + 1, :], elite[19 - i:19 - i + 1, :])

    newpopulation = np.zeros((30, 54))

    newpopulation[0:20] = elite
    newpopulation[20:30] = children

    return newpopulation

if __name__ == '__main__':

    iterations = 100

    for i in range(iterations):

        population = init()
        population = calculate(population)
        print("Longest path: ", max(population[:, 53]))
        print("Shortest path: ", min(population[:, 53]))
        elitepop = elitism(population)
        population = crossover(elitepop)
