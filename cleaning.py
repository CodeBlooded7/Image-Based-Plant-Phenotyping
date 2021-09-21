import os
from PIL import Image
import numpy as np
import random

data = {}
val_split = 0.16

for i in ['healthy', 'Early_blight', 'Late_blight']:
    tmp = []
    s = 'data/'+i+'/'
    entries = os.scandir(s)
    for entry in entries:
        im = Image.open(s+entry.name)
        tmp.append(np.asarray(im))
    data[i] = tmp
    random.shuffle(data[i])
    print(f"{i} done!")

train_x = []
val_x = []
train_y = []
val_y = []

lab = 0
for i in ['healthy', 'Early_blight', 'Late_blight']:
    c = 0
    while c < int(val_split*len(data[i])):
        val_x.append(data[i][c])
        val_y.append(lab)
        c = c + 1
    while c < len(data[i]):
        train_x.append(data[i][c])
        train_y.append(lab)
        c = c + 1
    lab = lab + 1


print(len(train_x))
print(train_x[0].shape)
print(len(val_x))
print(val_x[0].shape)


random.seed(10)
random.shuffle(train_x)
random.shuffle(val_x)
random.seed(10)
random.shuffle(train_y)
random.shuffle(val_y)

data_x = np.array(train_x)
print(data_x.shape)
print(data_x[0].shape)
data_x = data_x.reshape((len(train_x), 256, 256, 3))
data_y = np.array(train_y).reshape(len(train_y), 1)
val_x = np.array(val_x).reshape((len(val_x), 256, 256, 3))
val_y = np.array(val_y).reshape(len(val_y), 1)

print(data_y.shape)
print(val_y.shape)

np.save("train_x", data_x)
np.save("train_y", data_y)
np.save("val_x", val_x)
np.save("val_y", val_y)
