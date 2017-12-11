import numpy as np

train_count = np.zeros(100)
with open('new_train.txt', 'r') as f:
    for line in f:
        train_count[int(line.rsplit()[-1])] += 1
#print "trainseg", train_count
train_prob = train_count / np.sum(train_count)
print np.sort(-train_prob)[:20]
print np.argsort(train_prob)[:20]

val_count = np.zeros(100)
with open('new_val.txt', 'r') as f:
    for line in f:
        val_count[int(line.rsplit()[-1])] += 1
#print "valseg", val_count
val_count = val_count / 50.
val_prob = val_count / np.sum(val_count)
print np.sort(-val_prob)[:20]
print np.argsort(val_prob)[:20]

print -np.sum(val_prob*np.log((train_prob + 1e-8)/(val_prob + 1e-8)))

