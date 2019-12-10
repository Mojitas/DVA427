import numpy as np
import random as rng

data_array = np.array(pd.read_csv("berlin52.tsp", header=None))

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

        data_array(salesmen(j, i),
