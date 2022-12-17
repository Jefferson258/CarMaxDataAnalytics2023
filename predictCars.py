# predicts car to buy based on car sold to carmax

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import csv
import numpy as np
import re
from sklearn.model_selection import train_test_split

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

# gets rid of all rows with null values in any column
nulls = 0
for i in range(len(data)):
    for j in range(len(data[0])):
        try:
            if data[i][j] == 'null':
                del data[i]
                i = i - 1
                nulls = nulls + 1
        except(IndexError):
            continue
            # print(i)

print('DATA:', len(data))
# making the regressor
rfr = RandomForestRegressor()

features_indices = [16, 17, 19, 20, 22, 25, 26] #[0, 1, 3, 4, 6, 7, 9]
target_indices = [0, 3, 4, 6]

# get feature set of data
feature_set = []
for row in data:
    feature_set.append([row[i] for i in features_indices])

# get target set for training set
target_set = []
for row in data:
    target_set.append([row[i] for i in target_indices])

X_train, X_test, y_train, y_test = train_test_split(
                                    feature_set, target_set, test_size=0.2, random_state=42)

# print(X_test)
# print(y_test)
print('xtest', len(X_test))
print('xtrain', len(X_train))
print('ytrain', len(y_train))
print('ytest', len(y_test))
print('data', len(data))
print(nulls)

# rfr.fit(X_train, y_train)
