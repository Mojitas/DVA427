import numpy as np
import random as rnd
import pandas as pd
import math
import matplotlib.pyplot as plt

###Globals###



data_array = np.array(pd.read_csv("berlin52.tsp", header=None))
distance_array = np.zeros((52, 52))  # List for distances
###
data_array = data_array.astype(int)
distance_array = distance_array.astype((int))


# np.set_printoptions(threshold=np.inf)

class Genetics:

    def __init__(self):  # starting conditions
        self.amount = 100  # salesmen in each generation
        self.generations = 1000
        self.next_gen = 45  # salesmen in next generation
        self.children = np.zeros((95, 54))
        self.child = np.zeros((1, 54))
        self.uelite = np.zeros((5, 54))
        self.elite = np.zeros((self.next_gen, 54))
        self.unelite = np.zeros((50, 54))

    def dawn_of_time(self):
        self.salesmen = np.zeros((amount, 54))  # every salesman has a route of cities and total distance

        for i in range(1, 52):
            self.salesmen[:, i] = i  # set cities in order for all the salesmen

        for i in range(amount):
            rnd.shuffle(self.salesmen[i, 1:52])  # shuffle order of the cities

        self.salesmen[:, 52] = self.salesmen[:, 0]  # start is same as end

        return self.salesmen

    def ultra_elitism(self, salesmen):
        self.uelite = self.uelite.astype(int)

        for j in range(5):  # number of best ones to save

            temp_min = min(salesmen[:, 53])  # check for shortest path

        for i in range(self.amount):  # Check through all the salesmen

            if temp_min == salesmen[i, 53]:  # If we got a match for good result
                temp_mini = i

        self.uelite[j] = salesmen[temp_mini]  # Then transfer to the elites
        salesmen[temp_mini, 53] = 100000  # Set to big value to not find the same path again

        return uelite

    def elitism(self, salesmen):

        self.elite = self.elite.astype(int)
        temp_min = 0

        for i in range(self.next_gen):  # Find best paths of generation

            temp_min = min(salesmen[:, 53])

            for j in range(self.amount):
                if temp_min == salesmen[j, 53]:
                    self.elite[i] = salesmen[j]
                    salesmen[j, 53] = 100000
                    break  # stop if we found the right index and move on to the next.

        return elite

    def un_elitism(self, salesmen):  #

        self.unelite = self.unelite.astype(int)
        counter = 0

        for i in range(100):

            if salesmen[i, 53] != 100000:
                self.unelite[counter] = salesmen[i]
                counter += 1

        return self.unelite

    def new_start(self):
        return

    def cross(self, parent1, parent2):  # makes new salesmen

        self.child = self.child.astype(int)
        self.child[0, 0] = 1  # starting city

        remaining = np.zeros((1, 54))  #
        remaining = remaining.astype(int)

        random_amount = rnd.randint(10, 15)  # 10-15 genes
        random_position = rnd.randint(1, 52 - random_amount)

        self.child[0, random_position:random_position + random_amount] = parent1[0,
                                                                    random_position:random_position + random_amount]  # Get genes from parent

        empty = 0

        for i in range(53):  # Get remaining places

            if parent2[0, i] not in self.child:
                remaining[0, empty] = parent2[0, i]
                empty += 1

        j = 0

        for i in range(52):  # fill remaining places with genes from parent 2

            if self.child[0, i] == 0:
                self.child[0, i] = remaining[0, j]
                j += 1

        self.child[0, 52] = self.child[0, 0]

        return self.child

    def crossover(self, uelite, elite, unelite):  # sends parents to cross2
        np.random.shuffle(elite)
        np.random.shuffle(unelite)
        self.children = self.children.astype(int)

        for i in range(95):
            prob = rnd.randint(1, 10)
            parent1 = rnd.randint(1, 44)
            parent2 = rnd.randint(1, 44)

            if prob < 8:
                self.children[i] = self.cross(elite[parent1:parent1 + 1], elite[parent2:parent2 + 1])

            elif prob == 8 or prob == 9:
                self.children[i] = self.cross(unelite[parent1:parent1 + 1], elite[parent2:parent2 + 1])

            elif prob == 10:
                self.children[i] = self.cross(unelite[parent1:parent1 + 1], unelite[parent2:parent2 + 1])

        new_population = np.zeros((self.amount, 54))
        new_population = new_population.astype(int)

        new_population[0:95] = self.children
        new_population[95:self.amount] = uelite

        return new_population

    def mutate(self, salesmen):
        for j in range(self.amount):
            mutations = rnd.randint(1, 7)
            random_position = rnd.randint(1, 52 - mutations)

            reverse = salesmen[j, random_position:random_position + mutations]

            reverse = np.flip([reverse])[0]  # flip the sequence

            salesmen[j, random_position:random_position + mutations] = reverse

        return salesmen

class Distances:

    def __init__(self):
        self.x1=0
        self.x2=0
        self.y1=0
        self.y2=0

    # check how long the road for the salesmen is
    def lookup(self, salesman):
        salesman = salesman.astype(int)
        for i in range(Gen.amount):
            for j in range(52):
                salesman[i, 53] += distance_array[salesman[i, j] - 1, salesman[i, j + 1] - 1]

        return salesman

    # Takes the distance array
    def calculation(self, x):
        for j in range(52):  # run once to check the distances between all cities

            self.x1 = data_array[j, 1]  # compare every city to the rest
            self.y1 = data_array[j, 2]

            for i in range(52 - j):  # is for staying inside the matrix

                self.x2 = data_array[i + j, 1]  # +j for only populate half of the matrix
                self.y2 = data_array[i + j, 2]  # Since the values just are mirrored along the diagonal
                x[j, i + j] = math.sqrt((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)
        x += x.T
        return x

class Representation:
    #  plots stuff
    def __init__(self):
        self.x_list = []  # plot coordinates lists
        self.y_list = []

    def plot_data(self,x, y):
        plt.plot(x, y)
        plt.legend(('Best path'))
        plt.ylabel("Y")
        plt.xlabel("X")
     plt.show()


if __name__ == '__main__':

    Gen = Genetics()
    Dist = Distances()
    Rep=Representation()

    population = Gen.dawn_of_time()  # sets the first generation going
    shortest_path = 30000
    shortest_index = 0

    best_salesman = np.zeros((1, 54))
    distance_array = Dist.calculation(distance_array)

    for j in range(Gen.generations):

        population = Dist.lookup(population)

        if shortest_path > min(population[:, 53]):
            shortest_path = min(population[:, 53])  # assign the best one
            shortest_index = np.where(population == shortest_path)  # Find index of best one
            best_salesman = population[shortest_index[0]]

        if j % 20 == 0 and j > 0:  # print stuff

            print("Shortest so far: ", shortest_path)
            print("Average path: ", sum(population[:, 53]) / Gen.amount)

        ultraelitepop = Gen.ultra_elitism(population)
        elitepop = Gen.elitism(population)
        unelitepop = Gen.un_elitism(population)
        population = Gen.crossover(ultraelitepop, elitepop, unelitepop)
        population = Gen.mutate(population)
        population[:, 53] = 0  # resets everyones path

    print("City list: {}".format(best_salesman))

    for k in range(53):
        Rep.x_list.append(data_array[best_salesman[0, k] - 1, 1])
        Rep.y_list.append(data_array[best_salesman[0, k] - 1, 2])

    Rep.plot_data(Rep.x_list, Rep.y_list)
