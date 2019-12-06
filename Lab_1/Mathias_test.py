import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data transfer stuff, changed to 19 inputs as referenced by the pdf
data_array = np.array(pd.read_csv("assignment1.txt", header=None), dtype=np.float128)  # Reads from file
np.set_printoptions(formatter={'float': '{: 0.5f}'.format})  # formats to 5 decimal places
np.random.seed()
random.shuffle(data_array)  # Shuffles data

class dataManagement():

    def __init__(self): # Initialize the different arrays
        self.training_inputs = np.zeros((864, 19))
        self.validation_data = np.zeros((115, 19))
        self.test_data = np.zeros((172, 19))
        self.training_outputs = np.zeros((864, 1))
        self.validation_result = np.zeros((115,1))
        self.test_result = np.zeros((172,1))

    def data_normalization(self):

        #Uses the training data to normalize all of the other data.
        #As we would IRL

        for i in range(2,18):

            x1 = self.training_inputs[:,i]
            x_min = min(x1)
            x_max = max(x1)
            self.training_inputs[:,i] = (x1-x_min)/(x_max-x_min)
            x2 = self.validation_data[:, i]
            x3 = self.test_data[:, i]
            self.validation_data[:, i] = (x2 - x_min) / (x_max - x_min)
            self.test_data[:, i] = (x3 - x_min) / (x_max - x_min)


    # Puts the different data in the arrays
    def segmentation(self):


        self.training_inputs = data_array[0:864, 0:19]
        self.training_outputs = data_array[0:864, 19:20]

        # transfers validation set
        self.validation_data = data_array[864:979, 0:19]
        self.validation_result = data_array[864:979, 19:20]
        # transfers test set
        self.test_data = data_array[979:1151, 0:19]
        self.test_result = data_array[979:1151, 19:20]
        self.data_normalization()
        print("\nData has been managed\n")

DM=dataManagement() # Slippa skriva långa grejer
#