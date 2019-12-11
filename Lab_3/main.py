import numpy as np
import random as rng
import pandas as pd
import math

##
amount = 30  # Thirty salesmen in each generation

data_array = np.array(pd.read_csv("berlin52.tsp", header=None))
data_array = data_array.astype(int)


def init():  # starting conditions
    salesmen = np.zeros((amount, 54))  # every salesman has a route of cities and total distance

    for i in range(52):
        salesmen[:, i] = i + 1  # set cities in order for all the salesmen

    #salesmen[:, 52] = 1     # actually redundant but left until further inspection
    #salesmen = salesmen.astype(int)

    for i in range(amount):
        rng.shuffle(salesmen[i, 0:52])  # shuffle order of the cities

    salesmen[:, 52] = salesmen[:, 0] # start is same as end

    return salesmen


def calculate(salesmen):  # distance between two places
    salesmen = salesmen.astype(int)

    for j in range(amount):

        for i in range(52):  # for all salesmen
            x1 = data_array[salesmen[j, i] - 1, 1]
            y1 = data_array[salesmen[j, i] - 1, 2]  # pick coordinates of a city from data_array

            x2 = data_array[salesmen[j, i + 1] - 1, 1]
            y2 = data_array[salesmen[j, i + 1] - 1, 2]  # pick coordinates of the next city

            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)   # calculates the distance between the two cities

            salesmen[j, 53] += distance  # adds it to the total distance

    return salesmen


def elitism(salesmen):
    elite = np.zeros((20, 54))   # Variable for choosing choosing who gets to live
    elite = elite.astype(int)

    for j in range(20):  # number of best ones to save

        tempmin = min(salesmen[:, 53])  # check for shortest path

        for i in range(amount):     # Check through all the salesmen

            if tempmin == salesmen[i, 53]:  # If we got a match for good result
                tempmini = i

        elite[j] = salesmen[tempmini]   # Then transfer to the elites

        salesmen[tempmini, 53] = 100000     # Set to big value to solve some other problem I guess

    return elite


def cross(parent1, parent2):
    child = np.zeros((1, 54))
    child = child.astype(int)

    remaining = np.zeros((1, 54))
    remaining = remaining.astype(int)

    for j in range(10):

        randomtemp = rng.randint(0, 49)

        for i in range(4):
            child[0, randomtemp + i] = parent1[0, randomtemp + i]

    empty = 0

    for i in range(53):

        if parent2[0, i] not in child:
            remaining[0, empty] = parent2[0, i]
            empty += 1

    j = 0

    for i in range(52):

        if child[0, i] == 0:
            child[0, i] = remaining[0, j]
            j += 1;

    child[0, 52] = child[0, 0]

    return child


def crossover(elite):
    np.random.shuffle(elite)

    # print(elite[0, :])

    children = np.zeros((10, 54))
    children = children.astype(int)

    for i in range(10):
        children[i] = cross(elite[i:i + 1, :], elite[19 - i:19 - i + 1, :])

    # print(children[0])

    newpopulation = np.zeros((30, 54))
    newpopulation = newpopulation.astype(int)

    newpopulation[0:20] = elite
    newpopulation[20:30] = children

    newpopulation = newpopulation.astype(int)

    return newpopulation


def mutate(salesmen):
    for j in range(amount):

        for i in range(5):  # Arbitrary amount of random mutations

            random1 = rng.randint(0, 51)
            random2 = rng.randint(0, 51)

            temp = salesmen[j, random1]
            temp = temp.astype(int)

            salesmen[j, random1] = salesmen[j, random2]
            salesmen[j, random2] = temp

    salesmen[:, 52] = salesmen[:, 0]

    return salesmen


if __name__ == '__main__':

    iterations = 10000

    population = init()

    for i in range(iterations):

        population = calculate(population)

        if i % 100 == 0:        # print every
            print("Longest path: ", max(population[:, 53]))
            print("Shortest path: ", min(population[:, 53]))

        elitepop = elitism(population)
        population[:, :] = 0

        population = crossover(elitepop)
        population = mutate(population)

        population[:, 53] = 0
