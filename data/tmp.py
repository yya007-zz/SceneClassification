import numpy as np

train_count = np.zeros(100)
with open('new_train.txt', 'r') as f:
    for line in f:
        train_count[int(line.rsplit()[-1])] += 1
print "trainseg", train_count
train_count = train_count / 50.

val_count = np.zeros(100)
with open('new_val.txt', 'r') as f:
    for line in f:
        val_count[int(line.rsplit()[-1])] += 1
print "valseg", val_count
val_count = val_count / 50.


