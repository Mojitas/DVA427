import numpy as np
import pandas as pd

rate = 0.9      #Never used?
discount = 1

data_array = np.array(pd.read_csv("city 1.txt", header=None), dtype=(np.unicode_, 1))  # Reads from file
data_array[:, 2] = data_array[:, 2].astype(int) - discount  # what is discount?


class Node:
    def __init__(self, value=None):
        self.value = value
        self.connections = []  # list with connections
        self.weights = []  # list with connection weights


node_list = [Node('F')]

counter = 1  # ??

for C in range(23):
    for i in range(35):
        if data_array[i, 0] == node_list[C].value:
            node_list[C].connections.append(data_array[i, 1])
            node_list[C].weights.append(data_array[i, 2].astype(int))
        elif data_array[i, 1] == node_list[C].value:
            node_list[C].connections.append(data_array[i, 0])
            node_list[C].weights.append(data_array[i, 2].astype(int))

    for i in range(len(node_list[C].connections)):
        alreadyIn = 0
        letter = node_list[C].connections[i]

        for j in range(len(node_list)):
            if letter == node_list[j].value:
                alreadyIn = 1

        if alreadyIn == 0:
            node_list.append(Node(letter))

for i in range(len(node_list)):
    node_list[i].vstar = 100000

    if node_list[i].value == 'F':
        node_list[i].vstar = 0

for j in range(len(node_list)):
    print(node_list[j].value)

print(len(node_list))

# print(data_array)
