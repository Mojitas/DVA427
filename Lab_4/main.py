import numpy as np
import pandas as pd

rate = 0.9
discount = 1

data_array = np.array(pd.read_csv("city 1.txt", header=None), dtype=(np.unicode_, 1))  # Reads from file

for i in range(35):
    data_array[i, 2] = data_array[i, 2].astype(int) - discount


class Node:
    def __init__(self, value=None):

        self.value = value

        self.connections = [] #Array med connections
        self.weights = [] #Array med connectionsens weights


list = [Node('F')]

counter = 1

for C in range(23):
    for i in range(35):
        if data_array[i, 0] == list[C].value:
            list[C].connections.append(data_array[i, 1])
            list[C].weights.append(data_array[i, 2].astype(int))
        elif data_array[i, 1] == list[C].value:
            list[C].connections.append(data_array[i, 0])
            list[C].weights.append(data_array[i, 2].astype(int))

    for i in range(len(list[C].connections)):
        alreadyIn = 0
        letter = list[C].connections[i]

        for j in range(len(list)):
            if letter == list[j].value:
                alreadyIn = 1

        if alreadyIn == 0:
            list.append(Node(letter))

for i in range(len(list)):
    list[i].vstar = 100000

    if list[i].value == 'F':
        list[i].vstar = 0






for j in range(len(list)):
    print(list[j].value)

print(len(list))


#print(data_array)
