import numpy as np
import pandas as pd

data_array = np.array(pd.read_csv("city 1.txt", header=None), dtype=(np.unicode_, 1))  # Reads from file
data_array[:, 2] = data_array[:, 2].astype(int) - 1  # what is discount?


class Node:
    def __init__(self, value=None):
        self.value = value
        self.connections = []  # list with connections
        self.weights = []  # list with connection weights



exit('Butt!')
if __name__ == '__main__':

    node_list = [Node('F')]

    for i in range(23):
        for j in range(35):
            if data_array[j, 0] == node_list[i].value:
                node_list[i].connections.append(data_array[j, 1])
                node_list[i].weights.append(data_array[i, 2].astype(int))
            elif data_array[j, 1] == node_list[i].value:
                node_list[i].connections.append(data_array[j, 0])
                node_list[i].weights.append(data_array[j, 2].astype(int))

        for j in range(len(node_list[i].connections)):
            alreadyIn = 0
            letter = node_list[i].connections[j]

            for k in range(len(node_list)):
                if letter == node_list[k].value:
                    alreadyIn = 1

            if alreadyIn == 0:
                node_list.append(Node(letter))

    for i in range(len(node_list)):
        node_list[i].vstar = 100000

        if node_list[i].value == 'F':
            node_list[i].vstar = 0

    for i in range(len(node_list)):
        print(node_list[i].value)
