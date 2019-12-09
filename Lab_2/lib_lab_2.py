import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_array = np.array(pd.read_csv("assignment1.txt", header=None), dtype=np.float128)  # Reads from file
np.set_printoptions(formatter={'float': '{: 0.5f}'.format})  # formats to 5 decimal places
np.random.seed()
random.shuffle(data_array)  # Shuffles data


class data_management():

    def __init__(self):

    def segmentation(self):

    def normalization(self):

    def randomizing(self):
    


DM=data_management()



