from os import path
import pandas as pd

data = pd.read_csv('dataset/2017.csv')

a = data['path'].as_matrix()
count_true = 0
count_false = 0
for elt in a:
    if(path.exists(elt)):
        count_true = count_true +1
    else:
        count_false = count_false +1

print(count_true)
print(count_false)