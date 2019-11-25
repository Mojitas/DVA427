import numpy as np
import pandas as pd
import random

# Data transfer stuff, changed to 19 inputs as referenced by the pdf
data_array = np.array(pd.read_csv("assignment1.txt"))  # Reads from file
np.set_printoptions(formatter={'float': '{: 0.15f}'.format})  # formats to 3 decimal places
#random.shuffle(data_array)  # randomises the data, unneccesary at first
training_inputs = np.zeros((16, 54, 19))  # global variables for the data
validation_data = np.zeros((115, 19))
test_data = np.zeros((172, 19))
training_outputs = np.zeros((16, 54, 1))
validation_result = np.zeros(115)
test_result = np.zeros(172)
np.random.seed()
x=np.array([1,1,22,22,22,19,18,14,49.895756,17.775994,5.27092,0.771761,0.018632,0.006864,0.003923,0.003923,0.486903,0.100025,1],np.double)
r=0
y= 2 * np.random.random((19,1))-1

print(1 / (1 + np.exp(-np.dot(x.T,y))))

def data_segmentation():
    for i in range(16):  # transfers the training set
        for j in range(54):
            training_inputs[i, j, :] = data_array[j + i * 54, 0:19]
            training_outputs[i, j,:] = data_array[j + i * 54, 19]

    for i in range(864, 979):  # transfers validation set
        validation_data[i - 864, :] = data_array[i, 0:19]
        validation_result[i - 864] = data_array[i, 19]
    for i in range(979, 1151):  # transfers test set
        test_data[i - 979, :] = data_array[i, 0:19]
        test_result[i - 979] = data_array[i, 19]

    print("\nData has been segmented\n")


