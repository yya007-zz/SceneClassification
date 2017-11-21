import numpy as np 
import time

def save(result, address):
	name = []
	with open('../../data/test.txt', 'r') as f:
		for line in f:
			path, lab =line.rstrip().split(' ')
			name.append(path)

	file = open(address+".txt",'w')
	for i in range(10000):
		prediction = ""
		for x in result[i]:
			prediction += " "+str(x)
		file.write(name[i]+prediction+"\n")
	file.close()