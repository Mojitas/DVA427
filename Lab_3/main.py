import numpy as np
import random as rnd
import pandas as pd
import math
import matplotlib.pyplot as plt

###Globals###
amount = 100  # salesmen in each generation
data_array = np.array(pd.read_csv("berlin52.tsp", header=None))
distance_array = np.zeros((52, 52))  # List for distances
###
data_array = data_array.astype(int)


# np.set_printoptions(threshold=np.inf)


def init():  # starting conditions
    salesmen = np.zeros((amount, 54))  # every salesman has a route of cities and total distance

    for i in range(1, 52):
        salesmen[:, i] = i  # set cities in order for all the salesmen

    for i in range(amount):
        rnd.shuffle(salesmen[i, 1:52])  # shuffle order of the cities

    salesmen[:, 52] = salesmen[:, 0]  # start is same as end

    return salesmen


def ultra_elitism(salesmen):
    elite = np.zeros((5, 54))  # Variable for choosing choosing who gets to live
    elite = elite.astype(int)

    for j in range(5):  # number of best ones to save

        temp_min = min(salesmen[:, 53])  # check for shortest path

        for i in range(amount):  # Check through all the salesmen

            if temp_min == salesmen[i, 53]:  # If we got a match for good result
                temp_mini = i

        elite[j] = salesmen[temp_mini]  # Then transfer to the elites
        salesmen[temp_mini, 53] = 100000  # Set to big value to not find the same path again

    return elite


def elitism(salesmen):
    next_gen = 45  # salesmen in next generation
    elite = np.zeros((next_gen, 54))  # Variable for choosing choosing who gets to live
    elite = elite.astype(int)
    temp_min=0

    for i in range(next_gen):  # Find best paths of generation

        temp_min = min(salesmen[:, 53])

        for j in range(amount):
            if temp_min == salesmen[j, 53]:
                elite[i] = salesmen[j]
                salesmen[j, 53] = 100000
                break       #stop if we found the right index and move on to the next.

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

    random_amount = rnd.randint(10, 15)  # 10-15 genes
    random_position = rnd.randint(1, 52 - random_amount)

    child[0, random_position:random_position + random_amount] = parent1[0,
                                                                random_position:random_position + random_amount]  # Get genes from parent

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

    new_population = np.zeros((amount, 54))
    new_population = new_population.astype(int)

    new_population[0:95] = children
    new_population[95:amount] = ultraelite

    return new_population


def mutate(salesmen):
    for j in range(amount):
        mutations = rnd.randint(1, 7)
        random_position = rnd.randint(1, 52 - mutations)

        reverse = salesmen[j, random_position:random_position + mutations]

        reverse = np.flip([reverse])[0]  # flip the sequence

        salesmen[j, random_position:random_position + mutations] = reverse

    return salesmen


# check how long the road for the salesmen is
def distance_lookup(salesman):
    salesman = salesman.astype(int)
    for i in range(amount):
        for j in range(52):
            salesman[i, 53] += distance_array[salesman[i, j] - 1, salesman[i, j + 1] - 1]

    return salesman


# Takes the distance array
def distance_calculations(x):
    for j in range(52):  # run once to check the distances between all cities

        x1 = data_array[j, 1]  # compare every city to the rest
        y1 = data_array[j, 2]

        for i in range(52 - j):  # is for staying inside the matrix

            x2 = data_array[i + j, 1]  # +j for only populate half of the matrix
            y2 = data_array[i + j, 2]  # Since the values just are mirrored along the diagonal
            x[j, i + j] = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    x += x.T
    return x


#  plots stuff
def plot_data(x, y):
    plt.plot(x, y)
    plt.legend(('Best path'))
    plt.ylabel("Y")
    plt.xlabel("X")
    plt.show()


if __name__ == '__main__':

    generations = 1000  # number of generations
    population = init()  # sets the first generation going
    shortest_path = 30000
    shortest_index = 0
    x_list = []  # plot coordinates lists
    y_list = []
    best_salesman = np.zeros((1, 54))
    distance_array = distance_calculations(distance_array)

    for j in range(generations):

        population = distance_lookup(population)

        if shortest_path > min(population[:, 53]):
            shortest_path = min(population[:, 53])  # assign the best one
            shortest_index = np.where(population == shortest_path)  # Find index of best one
            best_salesman = population[shortest_index[0]]

        if j % 20 == 0 and j > 0:  # print stuff

            print("Shortest so far: ", shortest_path)
            print("Average path: ", sum(population[:, 53]) / amount)

        ultraelitepop = ultra_elitism(population)
        elitepop = elitism(population)
        unelitepop = un_elitism(population)
        population = crossover(ultraelitepop, elitepop, unelitepop)
        population = mutate(population)
        population[:, 53] = 0  # resets everyones path

    print("City list: {}".format(best_salesman))

    for k in range(53):
        x_list.append(data_array[best_salesman[0, k] - 1, 1])
        y_list.append(data_array[best_salesman[0, k] - 1, 2])

    plot_data(x_list, y_list)
