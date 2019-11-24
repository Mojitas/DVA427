import numpy as np
import pandas as pd

# TODO: Dela upp data i randomiserade grupper för träning, test och validering

x=np.array([0,1,2,3,4,5,6,7,8,9])

y=x[0:8]
z=x[8:10]

print("y: ",y)
print("z: ",z)

#Data transfer stuff
data_array = np.array(pd.read_csv("assignment1.txt"))
np.set_printoptions(formatter={'float': '{: 0.3f}'.format}) #formatterar utskrifter till max 3 decimaler

training_data = np.zeros((16,54,18))
validation_data = np.zeros((115,18))
test_data = np.zeros((172,18))
training_result = np.zeros((16,54,2))
validation_result = np.zeros((115,2))
test_result = np.zeros((172,2))

def data_transfer():
    for i in range(16): #transfers the training set
        for j in range(54):
            training_data[i,j,:]=data_array[j+i*54,0:18]
            training_result[i,j,:]=data_array[j+i*54,18:20]

    for i in range(864,979):         #transfers validation set
        validation_data[i-864,:] = data_array[i,0:18]
        validation_result[i-864,:] = data_array[i,18:20]
    for i in range(979,1151):    #transfers test set
        test_data[i-979,:] = data_array[i,0:18]
        test_result[i-979,:] = data_array[i,18:20]

    print("\nDone\n")
data_transfer()

print(test_result)