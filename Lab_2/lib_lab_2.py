import random
import numpy as np
import numpy as np
import random as rng
import pandas as pd

data_array = np.array(pd.read_csv("berlin52.tsp", header=None))

print(data_array)

amount = 20

salesmen = np.zeros((amount, 53))

for i in range(52):
    salesmen[:, i] = i + 1

salesmen[:, 52] = 1

for i in range(amount):

    rng.shuffle(salesmen[i])

print(salesmen[0, :])
print(salesmen[1, :])

for j in range(amount):
    for i in range(53):

        print(data_array[salesmen(j, i)])


data_array = np.array(pd.read_csv("iris.txt", header=None))  # Reads from file
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})  # formats to 5 decimal places
np.random.seed()
#random.shuffle(data_array)  # Shuffles data



class data_management():

    #def __init__(self):

    #def segmentation(self):

    def normalization(self):
        for i in range(4):
            xmax=max(data_array[:,i])
            xmin=min(data_array[:,i])
            data_array[:,i]=(data_array[:,i]-xmin)/(xmax-xmin)

    #def randomizing(self):



DM=data_management()
DM.normalization()




