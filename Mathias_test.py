import numpy as np
import pandas as pd
import math
import random

# Data transfer stuff, changed to 19 inputs as referenced by the pdf
data_array = np.array(pd.read_csv("assignment1.txt"))  # Reads from file
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})  # formats to 3 decimal places
#random.shuffle(data_array)  # randomises the data
training_inputs = np.zeros((16, 54, 19))  # global variables for the data
validation_data = np.zeros((115, 19))
test_data = np.zeros((172, 19))
training_outputs = np.zeros((16, 54))
validation_result = np.zeros(115)
test_result = np.zeros(172)


def data_segmentation():
    for i in range(16):  # transfers the training set
        for j in range(54):
            training_inputs[i, j, :] = data_array[j + i * 54, 0:19]
            training_outputs[i, j] = data_array[j + i * 54, 19]

    for i in range(864, 979):  # transfers validation set
        validation_data[i - 864, :] = data_array[i, 0:19]
        validation_result[i - 864] = data_array[i, 19]
    for i in range(979, 1151):  # transfers test set
        test_data[i - 979, :] = data_array[i, 0:19]
        test_result[i - 979] = data_array[i, 19]

    print("\nData has been segmented\n")
