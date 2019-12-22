import numpy as np
import pandas as pd


data_array = np.array(pd.read_csv("city 1.txt", header=None), dtype=(np.unicode_, 1))  # Reads from file


class Node:
    def __init__(self, value=None):

        self.value = value

        connections = 0; #Array med connections
        weights = 0; #Array med connectionsens weights

print(data_array)
