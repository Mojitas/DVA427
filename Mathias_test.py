import numpy as np
import pandas as pd

# TODO: Dela upp data i randomiserade grupper för träning, test och validering



y=np.array([[[1,2,3],[4,5,6],[7,8,9]],
            [[10,11,12],[13,14,15],[16,17,18]]])

x=np.array([[1,0,0],
            [0,1,0],
            [0,0,1]])

print(np.dot(y[0],x)) # det går att matrismultiplicera 3d och 2d arrayer

#Data transfer stuff
data_array = np.array(pd.read_csv("assignment1.txt"))
np.set_printoptions(formatter={'float': '{: 0.3f}'.format}) #formatterar utskrifter till max 3 decimaler

training_data = np.zeros((16,54,18))
validation_data = np.zeros((115,18))
test_data = np.zeros((171,18))
traning_result = np.zeros((16,54,2))
validation_result = np.array((115,2))
test_result = np.array((171,2))

def data_transfer():
    for i in range(16): #transfers the training set
        for j in range(54):
            training_data[i,j,:]=data_array[j+i*54,0:18]

    for i in range(864,979):         #transfers validation set
        validation_data[i-864,:]=data_array[i,0:18]
    for i in range(979,1150):    #transfers test set
        test_data[i-979,:]=data_array[i,0:18]

    print("\nDone\n")
data_transfer()