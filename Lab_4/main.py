import numpy as np
import pandas as pd

data_array = np.array(pd.read_csv("city 1.txt", header=None), dtype=(np.unicode_, 1))  # Reads from file
data_array[:, 2] = data_array[:, 2].astype(int) - 1  # what is discount?


class Node:
    def __init__(self, value=None):
        self.value = value
        self.connections = []  # list with connections
        self.weights = []  # list with connection weights

