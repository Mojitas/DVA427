import numpy as np
import pandas as pd

# TODO: Dela upp data i randomiserade grupper för träning, test och validering


data_array = np.array(pd.read_csv("assignment1.txt"))
np.set_printoptions(formatter={'float': '{: 0.3f}'.format}) #formatterar utskrifter till max 3 decimaler
print("Data: %a" %data_array[1,])
