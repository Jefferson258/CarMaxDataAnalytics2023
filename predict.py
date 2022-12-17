# predicts car to buy based on car sold to carmax

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import csv
import numpy as np
import re

# Open the CSV file and create a reader object
with open("ShowcaseDataWinter2023.csv", newline="") as csvfile:
    reader = csv.reader(csvfile)

    # Read the header row
    header = next(reader)

    # Create an empty list to store the data
    data = []

    # Loop through the rows in the reader object
    for row in reader:
        data.append(row)

# Create a LabelEncoder object
le = LabelEncoder()
for i in range(len(data[0])):
    column = [row[i] for row in data]
    if not (column[1].isdigit()):
        column = le.fit_transform(column)
        for j, row in enumerate(data):
            row[i] = column[j]


nulls = 0
for i in range(len(data)):
    for j in range(len(data[0])):
        try:
            data[i][j] = int(data[i][j])
        except(ValueError):
            del data[i]
            i = i - 1
            nulls = nulls + 1
        except(IndexError):
            break


rfr = RandomForestRegressor()

features_indices = [16, 17, 19, 20, 22, 25, 26] #[0, 1, 3, 4, 6, 7, 9]
target_indices = [0, 3, 4, 6]

# get training set of data
count = 0
training_set = []
for row in data:
    if count > 100000:
        break
    training_set.append([row[i] for i in features_indices])
    count  = count + 1

# get testing set of data
count = 0
testing_set = []
for row in data:
    if count < 100000:
        count = count + 1
        continue
    else:
        testing_set.append([row[i] for i in features_indices])

# get target set for training set
training_target_set = []
count = 0
for row in data:
    if count > 100000:
        break
    training_target_set.append([row[i] for i in target_indices])
    count  = count + 1

# get target set for testing set
count = 0
testing_target_set = []
for row in data:
    if count < 100000:
        count = count + 1
        continue
    else:
        testing_target_set.append([row[i] for i in target_indices])

print(len(training_set))
print('Test:', len(testing_set))

rfr.fit(training_set, training_target_set)

print('testing')
test = np.array(testing_set[2])
print(rfr.predict(test.reshape(1, -1)))
print(testing_target_set[2])

print('TEST2')
test = np.array(testing_set[200])
print(rfr.predict(test.reshape(1, -1)))
print(testing_target_set[200])

# print(data)
print(len(data))
print(len(data[0]))
print(nulls)