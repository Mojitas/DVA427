import numpy as np
import random as rnd
import pandas as pd
import math
import matplotlib.pyplot as plt

amount = 100  # salesmen in each generation

data_array = np.array(pd.read_csv("berlin52.tsp", header=None))
np.set_printoptions(threshold=np.inf)
data_array = data_array.astype(int)


def init():  # starting conditions
    salesmen = np.zeros((amount, 54))  # every salesman has a route of cities and total distance

    for i in range(52):
        salesmen[:, i] = i + 1  # set cities in order for all the salesmen

    for i in range(amount):
        rnd.shuffle(salesmen[i, 1:52])  # shuffle order of the cities

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


def ultra_elitism(salesmen):
    elite = np.zeros((5, 54))  # Variable for choosing choosing who gets to live
    elite = elite.astype(int)

    for j in range(5):  # number of best ones to save

        tempmin = min(salesmen[:, 53])  # check for shortest path

        for i in range(amount):  # Check through all the salesmen

            if tempmin == salesmen[i, 53]:  # If we got a match for good result
                tempmini = i

        elite[j] = salesmen[tempmini]  # Then transfer to the elites

        salesmen[tempmini, 53] = 100000  # Set to big value to not find the same path again

    return elite


def elitism(salesmen):
    elite = np.zeros((45, 54))  # Variable for choosing choosing who gets to live
    elite = elite.astype(int)

    for j in range(45):  # number of best ones to save

        tempmin = min(salesmen[:, 53])  # check for shortest path

        for i in range(amount):  # Check through all the salesmen

            if tempmin == salesmen[i, 53]:  # If we got a match for good result
                tempmini = i

        elite[j] = salesmen[tempmini]  # Then transfer to the elites
        salesmen[tempmini, 53] = 100000  # Set to big value to not find the same path again

    return elite


def un_elitism(salesmen):  #

    unelite = np.zeros((50, 54))
    unelite = unelite.astype(int)
    counter = 0

    for i in range(100):

        if salesmen[i, 53] != 100000:
            unelite[counter] = salesmen[i]
            counter += 1

    return unelite


def cross(parent1, parent2):  # makes new salesmen
    child = np.zeros((1, 54))
    child = child.astype(int)

    child[0, 0] = 1  # starting city

    remaining = np.zeros((1, 54))  #
    remaining = remaining.astype(int)

    randamount = rnd.randint(10, 15)  # 10-15 genes
    randompos = rnd.randint(1, 52 - randamount)

    child[0, randompos:randompos + randamount] = parent1[0, randompos:randompos + randamount]  # Get genes from parent

    empty = 0

    for i in range(53):  # Get remaining places

        if parent2[0, i] not in child:
            remaining[0, empty] = parent2[0, i]
            empty += 1

    j = 0

    for i in range(52):  # fill remaining places with genes from parent 2

        if child[0, i] == 0:
            child[0, i] = remaining[0, j]
            j += 1

    child[0, 52] = child[0, 0]

    return child


def crossover(ultraelite, elite, unelite):  # sends parents to cross2
    np.random.shuffle(elite)
    np.random.shuffle(unelite)

    children = np.zeros((95, 54))
    children = children.astype(int)

    for i in range(95):
        prob = rnd.randint(1, 10)
        parent1 = rnd.randint(1, 44)
        parent2 = rnd.randint(1, 44)

        if prob < 8:
            children[i] = cross(elite[parent1:parent1 + 1], elite[parent2:parent2 + 1])

        elif prob == 8 or prob == 9:
            children[i] = cross(unelite[parent1:parent1 + 1], elite[parent2:parent2 + 1])

        elif prob == 10:
            children[i] = cross(unelite[parent1:parent1 + 1], unelite[parent2:parent2 + 1])

    newpopulation = np.zeros((100, 54))
    newpopulation = newpopulation.astype(int)

    newpopulation[0:95] = children
    newpopulation[95:100] = ultraelite

    return newpopulation


def mutate(salesmen):
    for j in range(amount):
        mutations = rnd.randint(1, 7)
        randompos = rnd.randint(1, 52 - mutations)

        reverse = salesmen[j, randompos:randompos + mutations]

        reverse = np.flip([reverse])[0]  # flip the sequence

        salesmen[j, randompos:randompos + mutations] = reverse

    return salesmen


if __name__ == '__main__':

    generations = 101  # number of generations
    population = init()  # sets the first generation going
    lowest = 30000
    average2 = 0
    x_list = []  # plot lists
    y_list = []
    best_salesman = 0
    average = 0

    for R in range(generations):

        population = calculate(population)

        lowest = min(lowest, min(population[:, 53]))  # assign the best one
        best_salesman = np.where(population == lowest)  # Find the best one
        average += sum(population[:, 53]) / 100

        if R % 25 == 0 and R > 0:  # print every 25

            average = average / 25
            print("Average path: ", average)
            average = 0

        if R % 100 == 0:
            print("Shortest found path: ", lowest)

        ultraelitepop = ultra_elitism(population)
        elitepop = elitism(population)
        unelitepop = un_elitism(population)

        population = crossover(ultraelitepop, elitepop, unelitepop)
        population = mutate(population)
        population[:, 53] = 0  # resets everyones path

    print("Shortest path: {}\n City list: {}".format(lowest, best_salesman))

"""
for i in range(52):

    x_list.append(data_array[shortest_path[i],1])
    y_list.append(data_array[shortest_path[i],2])

plt.plot(x_list, y_list)
plt.legend(('Best path'))
plt.ylabel("Y")
plt.xlabel("X")
plt.show()

"""
