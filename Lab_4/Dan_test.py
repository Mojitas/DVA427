import numpy as np
import pandas as pd

data_array = np.array(pd.read_csv("city 1.txt", header=None), dtype=(np.unicode_, 1))

Paths = np.zeros((23, 10))
Paths[:, :] = -1
Paths[5, 0] = 5


class Graph:

    def __init__(self, nodes):
        self.Nodes = nodes
        self.graph = []

    def addEdge(self, city1, city2, weight):
        self.graph.append([city1, city2, weight])

    def printArr(self, dist):
        print([(chr(i+65), dist[i]) for i in range(self.Nodes)])

    def BellmanFord(self, src):

        dist = [float("Inf")] * self.Nodes
        dist[src] = 0

        for i in range(self.Nodes - 1):
            for u, v, w in self.graph:
                if dist[u] != [float("Inf")] and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    #print("Distance to " + str(v) + " has been updated via " + str(u))

                    Paths[v, :] = 0
                    Paths[v, :] = Paths[u, :]

                    stop = 0
                    j = 0

                    while stop == 0:
                        j = j + 1
                        if Paths[v, j] == -1: stop = 1

                    Paths[v, j] = v

        self.printArr(dist)

G = Graph(23)  # init the graph with 23 nodes
for i in range(35):  # Add the vertices(as numbers) and cost of edges
    G.addEdge(ord(data_array[i, 0]) - 65, ord(data_array[i, 1]) - 65, data_array[i, 2].astype(int))
for i in range(35):  # Undirected graph, each edge goes both ways
    G.addEdge(ord(data_array[i, 1]) - 65, ord(data_array[i, 0]) - 65, data_array[i, 2].astype(int))

G.BellmanFord(5)  # Find 5, aka F




for i in range(23):

    stop = 0
    j = 0

    while stop == 0:
        j = j + 1
        if Paths[i, j] == -1: stop = 1

    tempArray = Paths[i, 0:j]
    tempArray2 = []

    for k in range(len(tempArray)):
        tempArray2.append(chr((tempArray[k] + 65).astype(int)))

    print(tempArray2)

