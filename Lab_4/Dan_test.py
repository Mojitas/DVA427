import numpy as np
import pandas as pd

data_array = np.array(pd.read_csv("city 1.txt", header=None), dtype=(np.unicode_, 1))  # Reads from file


class Graph:

    def __init__(self, nodes):
        self.Nodes = nodes
        self.graph = []

    def addEdge(self, city1, city2, weigth):
        self.graph.append([city1, city2, weigth])

    def printArr(self, dist):
        print("Vertex Distance from Source")
        for i in range(self.Nodes):
            print(i, dist[i])

    def BellmanFord(self, src):

        dist = [float("Inf")] * self.Nodes
        dist[src] = 0

        for i in range(2):
            for u, v, w in self.graph:
                if dist[u] != [float("Inf")] and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w

        self.printArr(dist)


g = Graph(23)
for i in range(35):
    g.addEdge(ord(data_array[i, 0]) - 65, ord(data_array[i, 1]) - 65, data_array[i, 2].astype(int))
for i in range(35):  # Two sided?
    g.addEdge(ord(data_array[i, 1]) - 65, ord(data_array[i, 0]) - 65, data_array[i, 2].astype(int))

g.BellmanFord(5)