import numpy as np
import pandas as pd

data_array = np.array(pd.read_csv("city 1.txt", header=None), dtype=(np.unicode_, 1))
path = [5, 15]
node_list = [path for i in range(23)]  # a bit cleaner :)
print(node_list)


class Graph:

    def __init__(self, nodes):
        self.Nodes = nodes
        self.graph = []

    def addEdge(self, city1, city2, weight):
        self.graph.append([city1, city2, weight])

    def printArr(self, dist):
        print([(i, dist[i]) for i in range(self.Nodes)])

    def BellmanFord(self, src):

        dist = [float("Inf")] * self.Nodes
        dist[src] = 0

        for i in range(2):
            for u, v, w in self.graph:
                if dist[u] != [float("Inf")] and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    print("Distance to " + str(v) + " has been updated via " + str(u))

        self.printArr(dist)


G = Graph(23)
for i in range(35):
    G.addEdge(ord(data_array[i, 0]) - 65, ord(data_array[i, 1]) - 65, data_array[i, 2].astype(int))
for i in range(35):  # Two sided?
    G.addEdge(ord(data_array[i, 1]) - 65, ord(data_array[i, 0]) - 65, data_array[i, 2].astype(int))

G.BellmanFord(5)
