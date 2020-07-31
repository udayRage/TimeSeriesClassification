import numpy as np
import time
import math
#from scipy.spatial.distance import directed_hausdorff
def hausdorff(u,v):
    row, = u.shape
    lea_distance = 0
    for i in range(row):
        #for j in range(column):
        distance1 = np.amin(np.absolute((np.diff(u[i]-v))))
        if distance1 > lea_distance:
            lea_distance = distance1
    return lea_distance

import sys
training = np.loadtxt(sys.argv[1], delimiter='\t')
testing = np.loadtxt(sys.argv[2],  delimiter='\t')

num_rows_test, num_columns_test = testing.shape
num_rows_train, num_columns_train = training.shape
training_noclass=training[:,1:]
testing_noclass=testing[:,1:]

predicted_label = None
correct = 0
start_time = time.time()
for i in range(num_rows_test):
    least_distance = float('inf')
    for j in range(num_rows_train):
        dist = max(hausdorff(testing_noclass[i], training_noclass[j]), hausdorff(training_noclass[j], testing_noclass[i]))  # math.sqrt(squaring)
        if dist < least_distance:
            predicted_label = training[j][0]
            least_distance = dist
    if predicted_label == testing[i][0]:
        correct = correct + 1


accuracy = (correct/num_rows_test) * 100
print("Datasetname:",sys.argv[1])

print("Total Accuracy of oneNNHausdorff is:", accuracy)

print("Total Execution time of oneNNHausdorff", time.time() - start_time)
import os
import psutil
process = psutil.Process(os.getpid())
memory = process.memory_full_info().uss
memory_in_KB = memory /(1024)
print("Total Memory of oneNNHausdorff inKB",memory_in_KB)  # in bytes
