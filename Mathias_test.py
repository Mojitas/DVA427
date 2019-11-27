import random
import math
import numpy as np
import pandas as pd
# TODO Normalisera datat med en funktion
# Data transfer stuff, changed to 19 inputs as referenced by the pdf
data_array = np.array(pd.read_csv("assignment1.txt", header=None), dtype=np.float128)  # Reads from file
np.set_printoptions(formatter={'float': '{: 0.15f}'.format})  # formats to 3 decimal places
#np.random.seed()
random.shuffle(data_array)  # randomises the data, unneccesary at first

class dataManagement():

    def __init__(self):
        self.training_inputs = np.zeros((16, 54, 19))  # global variables for the data
        self.validation_data = np.zeros((115, 19))
        self.test_data = np.zeros((172, 19))
        self.training_outputs = np.zeros((16, 54, 1))
        self.validation_result = np.zeros((115,1))
        self.test_result = np.zeros((172,1))

    def data_normalization(self):

#Uses the traning data to normalize all of the other data.

        for i in range(2,18):

            for j in range(16):

                x1 = self.training_inputs[j,:,i]
                x_min = min(x1)
                x_max = max(x1)
                self.training_inputs[j,:,i] = (x1-x_min)/(x_max-x_min)

            x2 = self.validation_data[:, i]
            x3 = self.test_data[:, i]
            self.validation_data[:, i] = (x2 - x_min) / (x_max - x_min)
            self.test_data[:, i] = (x3 - x_min) / (x_max - x_min)



    def segmentation(self):

        for i in range(16):  # transfers the training set
            for j in range(54):
                self.training_inputs[i, j, :] = data_array[j + i * 54, 0:19]
                self.training_outputs[i, j,:] = data_array[j + i * 54, 19]

        for i in range(864, 979):  # transfers validation set
            self.validation_data[i - 864, :] = data_array[i, 0:19]
            self.validation_result[i - 864] = data_array[i, 19]
        for i in range(979, 1151):  # transfers test set
            self.test_data[i - 979, :] = data_array[i, 0:19]
            self.test_result[i - 979] = data_array[i, 19]
        self.data_normalization()
        print("\nData has been managed\n")

DM=dataManagement()

#x = np.array([[0,1,2,3,4,5,6,7,8,9]])
#for i in range(2,10):
#    print("x: ", x[:,i],x.shape)

