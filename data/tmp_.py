from random import shuffle
d = {}
num = 10
with open('val.txt', 'r') as f:
    for line in f:
        path, _class = line.rsplit()
        if _class in d.keys():
            d[_class].append(path)
        else:
            d[_class] = [path]

path_class_list = []
for key in d.keys():
    path_list = d[key]
    shuffle(path_list)
    for i in range(num):
        path_class_list.append((path_list[i], key))
shuffle(path_class_list)
with open('small_val.txt', 'w') as f:
    for path_class in path_class_list:
        f.write(path_class[0] + ' ' + str(path_class[1]) + '\n')   
            
