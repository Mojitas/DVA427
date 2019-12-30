import numpy as np
import pandas as pd

data_array = np.array(pd.read_csv("city 1.txt", header=None), dtype=(np.unicode_, 1))

node = []
node_list = pd.DataFrame(node for i in range(23))


print(node_list)


exit(1)
class Graph:

    def __init__(self, nodes):
        self.Nodes = nodes
        self.graph = []

    def addEdge(self, city1, city2, weight):
        self.graph.append([city1, city2, weight])

    def printArr(self, dist):
        print([(chr(i+65), dist[i]) for i in range(self.Nodes)])

    def BellmanFord(self, src):

        dist = [float("Inf")] * self.Nodes  # list that keeps track of distance to each node from F, init to infinity
        dist[src] = 0  # start at F
        print(dist)
        for j in range(2):  # v - 1 times in worst case
            for u, v, w in self.graph:  # u,v are cities and w is distance
                if dist[u] != [float("Inf")] and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w  # update if shorter path is found
                    print("Distance to " + chr(v+65) + " has been updated via " + chr(u+65))

        self.printArr(dist)


G = Graph(23)   # init the graph with 23 nodes
for i in range(35):     # Add the vertices(as numbers) and cost of edges
    G.addEdge(ord(data_array[i, 0]) - 65, ord(data_array[i, 1]) - 65, data_array[i, 2].astype(int))
for i in range(35):     # Undirected graph, each edge goes both ways
    G.addEdge(ord(data_array[i, 1]) - 65, ord(data_array[i, 0]) - 65, data_array[i, 2].astype(int))

G.BellmanFord(5)    # Find 5, aka F
