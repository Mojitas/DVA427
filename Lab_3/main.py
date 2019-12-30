import numpy as np
import random
import pandas as pd
import math
import matplotlib.pyplot as plt
import time

# Globals #
data_array = np.array(pd.read_csv("berlin52.tsp", header=None))
distance_array = np.zeros((52, 52))  # Array for distances
###########
data_array = data_array.astype(int)
distance_array = distance_array.astype(int)
random.seed()
np.set_printoptions(threshold=np.inf)


class Genetics:

    def __init__(self):  # starting conditions
        self.amount = 100  # salesmen in each generation
        self.generations = 10000
        self.next_gen = 45  # Number of elites
        self.mutation_chance = 0

        self.best_salesman = np.zeros((1, 54))
        self.salesmen = np.zeros((self.amount, 54))  # every salesman has a route of cities and total distance
        self.child = np.zeros((1, 54))
        self.children = np.zeros((95, 54))
        self.elite = np.zeros((self.next_gen, 54))
        self.uelite = np.zeros((5, 54))
        self.unelite = np.zeros((50, 54))

    def dawn_of_time(self):

        for i in range(52):
            self.salesmen[:, i] = i + 1  # set cities in order for all the salesmen

        for i in range(self.amount):
            random.shuffle(self.salesmen[i, 1:52])  # shuffle order of the cities

        self.salesmen[:, 52] = self.salesmen[:, 0]  # start is same as end

        return self.salesmen

    @staticmethod
    def new_start(population):

        ultraelitepop = Gen.ultra_elitism(population)
        elitepop = Gen.elitism(population)
        unelitepop = Gen.un_elitism(population)
        population = Gen.crossover(ultraelitepop, elitepop, unelitepop)
        population = Gen.mutate(population)
        population[:, 53] = 0  # resets everyones path
        return population

    def ultra_elitism(self, salesmen):
        self.uelite = self.uelite.astype(int)

        for i in range(5):  # number of best ones to save

            temp_min = min(salesmen[:, 53])  # check for shortest path
            for j in range(self.amount):  # Check through all the salesmen
                if temp_min == salesmen[j, 53]:  # If we got a match for good result
                    self.uelite[i] = salesmen[j]  # Then transfer to the elites
                    salesmen[j, 53] = 100000  # Set to big value to not find the same path again

        return self.uelite

    def elitism(self, salesmen):

        self.elite = self.elite.astype(int)

        for i in range(self.next_gen):  # number of best ones to save

            temp_min = min(salesmen[:, 53])  # check for shortest path

            for j in range(self.amount):  # Check through all the salesmen

                if temp_min == salesmen[j, 53]:  # If we got a match for good result
                    self.elite[i] = salesmen[j]  # Then transfer to the elites
                    salesmen[j, 53] = 100000  # Set to big value to not find the same path again

        return self.elite

    def un_elitism(self, salesmen):  #

        self.unelite = self.unelite.astype(int)
        counter = 0

        for i in range(100):

            if salesmen[i, 53] != 100000:
                self.unelite[counter] = salesmen[i]
                counter += 1

        return self.unelite

    def cross(self, parent_1, parent_2):  # makes new salesmen

        self.child[:, :] = 0
        self.child = self.child.astype(int)
        self.child[0, 0] = 1  # starting city

        remaining = np.zeros((1, 54))  #
        remaining = remaining.astype(int)

        random_amount = random.randint(10, 30)  # 10-15 genes
        random_position = random.randint(1, 52 - random_amount)

        self.child[0, random_position:random_position + random_amount] = parent_1[0,
                                                                         random_position:random_position + random_amount]  # Get genes from parent

        empty = 0

        for i in range(53):  # Get remaining places

            if parent_2[0, i] not in self.child:
                remaining[0, empty] = parent_2[0, i]
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
        self.children[:, :] = 0
        self.children = self.children.astype(int)

        for i in range(95):
            prob = random.randint(1, 10)
            parent1 = random.randint(1, 44)
            parent2 = random.randint(1, 44)
            u_parent = random.randint(1, 4)

            if prob < 5:
                self.children[i] = self.cross(elite[parent1:parent1 + 1], elite[parent2:parent2 + 1])

            elif 5 <= prob <= 6:
                self.children[i] = self.cross(unelite[parent1:parent1 + 1], elite[parent2:parent2 + 1])

            else:
                self.children[i] = self.cross(elite[parent1:parent1 + 1], uelite[u_parent:u_parent + 1])

        new_population = np.zeros((self.amount, 54))
        new_population = new_population.astype(int)

        new_population[0:95] = self.children
        new_population[95:self.amount] = uelite

        return new_population

    def mutate(self, salesmen):
        for j in range(self.amount):

            if np.random.rand(1) <= self.mutation_chance:
                mutations = random.randint(1, 30)
                random_position = random.randint(1, 52 - mutations)
                reverse = salesmen[j, random_position:random_position + mutations]
                reverse = np.flip([reverse])[0]  # flip the sequence
                salesmen[j, random_position:random_position + mutations] = reverse

        return salesmen


class Distances:

    def __init__(self):
        self.x1 = 0
        self.x2 = 0
        self.y1 = 0
        self.y2 = 0

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
        self.gen_list = []
        self.score_list = []

    def plot_data(self, x, y, x_label, y_label, title):
        plt.plot(x, y)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.title(title)
        plt.show()


Gen = Genetics()
Dist = Distances()
Rep = Representation()  # iniializes everything
population = Gen.dawn_of_time()  # sets the first generation going
Dist.calculation(distance_array)    # Makes a table of distances

if __name__ == '__main__':

    shortest_path = 40000  # some high starting value
    shortest_index = 0   # Index for finding shortest path
    latest_improvement = 0

    start = time.time()  # Times the GA
    for j in range(Gen.generations):

        population = Dist.lookup(population)  # checks the length of everyone's path

        if shortest_path > min(population[:, 53]):
            shortest_path = min(population[:, 53])  # assign the best one
            for l in range(100):
                if shortest_path == population[l, 53]:  # find the best one
                    shortest_index = l
                    break

            Gen.best_salesman = population[shortest_index]
            Rep.score_list.append(shortest_path)
            Rep.gen_list.append(j)
            latest_improvement = j
            print("\nNew shortest path: {}\nGeneration: {}".format(shortest_path, j))

        population = Gen.new_start(population)  # makes the next generation

        if j - latest_improvement > 100 and shortest_path < 9000:  # stops faster
            print("No improvements for 100 generations")
            print("Final generation: {}\nFinal mutation chance: {}".format(j, Gen.mutation_chance))
            print("Best path: ", Gen.best_salesman[0:53])
            break
        elif j - latest_improvement > 50 and Gen.mutation_chance < 0.10:  # increase mutation as we go along without improvement
            Gen.mutation_chance += 0.005

    end = time.time()  # Stops time

    for m in range(53):  # Puts the best path in a list
        Rep.x_list.append(data_array[Gen.best_salesman[m] - 1, 1])
        Rep.y_list.append(data_array[Gen.best_salesman[m] - 1, 2])

    print("Execution time of GA: ", end - start)
    Rep.plot_data(Rep.x_list, Rep.y_list, "Distance(x)", "Distance(y)", "Best total path")
    Rep.plot_data(Rep.gen_list, Rep.score_list, "Generations", "Path length", "Evolution of salesmen")

