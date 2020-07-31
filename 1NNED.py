
import numpy as np
import math
import time
import psutil
import os
import sys

start_time = time.time()
training = np.loadtxt(sys.argv[1], delimiter='\t')
testing = np.loadtxt(sys.argv[2],  delimiter='\t')

num_rows_test, num_columns_test = testing.shape
num_rows_train, num_columns_train = training.shape


predicted_label = None
correct = 0
#start_time = time.time()
for i in range(num_rows_test):
    least_distance = float('inf')
    for j in range(num_rows_train):
        squaring =0
        for k in range(num_columns_train-1):
            squaring = squaring+(testing[i][k+1] - training[j][k+1])**2
        dist = math.sqrt(squaring)
        if dist < least_distance:
            predicted_label = training[j][0]
            least_distance = dist
    if predicted_label == testing[i][0]:
        correct = correct + 1


accuracy = (correct/num_rows_test) * 100
#gc.collect()
print("Datasetname:",sys.argv[1])
print("Total Accuracy of 1NNED is:", accuracy)


print("Total Execution time 1NNED is:", time.time() - start_time)

process = psutil.Process(os.getpid())
memory = process.memory_full_info().uss
memory_in_KB = memory /(1024)
print("Memory of 1NNED in KB",memory_in_KB)  # in bytes
