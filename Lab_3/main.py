import numpy as np
import random as rng
import pandas as pd
import math

##
amount = 100  # salesmen in each generation

data_array = np.array(pd.read_csv("berlin52.tsp", header=None))
data_array = data_array.astype(int)


def init():  # starting conditions
    salesmen = np.zeros((amount, 54))  # every salesman has a route of cities and total distance

    for i in range(52):
        salesmen[:, i] = i + 1  # set cities in order for all the salesmen

    # salesmen[:, 52] = 1     # actually redundant but left until further inspection
    # salesmen = salesmen.astype(int)

    for i in range(amount):
        rng.shuffle(salesmen[i, 1:52])  # shuffle order of the cities

    salesmen[:, 52] = salesmen[:, 0]  # start is same as end

    return salesmen


def calculate(salesmen):  # distance between two places
    salesmen = salesmen.astype(int)

    for j in range(amount):  # for all salesmen

        for i in range(52):
            x1 = data_array[salesmen[j, i] - 1, 1]
            y1 = data_array[salesmen[j, i] - 1, 2]  # pick coordinates of a city from data_array

            x2 = data_array[salesmen[j, i + 1] - 1, 1]
            y2 = data_array[salesmen[j, i + 1] - 1, 2]  # pick coordinates of the next city

            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # calculates the distance between the two cities

            salesmen[j, 53] += distance  # adds it to the total distance

    return salesmen


def elitism(salesmen):
    elite = np.zeros((50, 54))  # Variable for choosing choosing who gets to live
    elite = elite.astype(int)

    for j in range(50):  # number of best ones to save

        tempmin = min(salesmen[:, 53])  # check for shortest path

        for i in range(amount):  # Check through all the salesmen

            if tempmin == salesmen[i, 53]:  # If we got a match for good result
                tempmini = i

        elite[j] = salesmen[tempmini]  # Then transfer to the elites

        salesmen[tempmini, 53] = 100000  # Set to big value to not find the same path again

    return elite


def cross(parent1, parent2):
    child = np.zeros((1, 54))  # new salesmen
    child = child.astype(int)

    child[0, 0] = 1  # starting city

    remaining = np.zeros((1, 54))  #
    remaining = remaining.astype(int)

    randamount = rng.randint(10, 15)  # 15-20 gener
    randompos = rng.randint(1, 52 - randamount)

    for i in range(randamount):  # Get
        child[0, randompos + i] = parent1[0, randompos + i]  # Get genes from parent

    empty = 0

    for i in range(53):

        if parent2[0, i] not in child:
            remaining[0, empty] = parent2[0, i]
            empty += 1

    j = 0

    for i in range(52):

        if child[0, i] == 0:
            child[0, i] = remaining[0, j]
            j += 1

    child[0, 52] = child[0, 0]

    return child


def crossover(elite):
    np.random.shuffle(elite)

    children = np.zeros((amount, 54))
    children = children.astype(int)

    for j in range(4):
        np.random.shuffle(elite)
        for i in range(25):
            children[25 * j + i] = cross(elite[i:i + 1, :], elite[49 - i:49 - i + 1, :])

    return children


def mutate(salesmen):
    for j in range(amount):

        mutations = rng.randint(1, 7)

        reverse = np.zeros((1, mutations))

        random1 = rng.randint(1, 52 - mutations)

        for i in range(mutations):
            reverse[0, mutations - 1 - i] = salesmen[j, random1 + i]

        for i in range(mutations):
            salesmen[j, random1 + i] = reverse[0, i]

    salesmen[:, 52] = salesmen[:, 0]

    return salesmen


if __name__ == '__main__':

    iterations = 10000  # generations

    population = init()

    lowest = 30000
    average2 = 0

    for R in range(iterations):

        population = calculate(population)

        lowest = min(lowest, min(population[:, 53]))

        average = sum(population[:, 53]) / 100

        average2 += average

        if R % 50 == 0:  # print every 100

            if R != 0:
                average2 = average2 / 50

            print("Average path: ", average2)

            average2 = 0

            # print("Longest path: ", max(population[:, 53]))
            # print("Shortest path: ", min(population[:, 53]))

        if R % 200 == 0:
            print("Shortest found path: ", lowest)

        elitepop = elitism(population)
        population[:, :] = 0

        population = crossover(elitepop)
        population = mutate(population)

        population[:, 53] = 0

    print("Shortest found path: ", lowest)
