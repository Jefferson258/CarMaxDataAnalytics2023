# predicts car to buy based on car sold to carmax

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
import csv
import re
import matplotlib.pyplot as plt


def average_values(s):
    # print(s)
    # Split the string into a list of substrings separated by "to" or "-"
    substrings = s.split("to")
    if len(substrings) == 1:
        substrings = s.split("-")

    # If there is only one value, return it as a float
    if len(substrings) == 1:
        # Remove all non-numeric characters from the string
        value = re.sub(r'[^\d.]', '', substrings[0])
        return float(value)

    # Otherwise, average the two values and return the result
    else:
        # Remove all non-numeric characters from the first string
        value1 = re.sub(r'[^\d.]', '', substrings[0])
        # Remove all non-numeric characters from the second string
        value2 = re.sub(r'[^\d.]', '', substrings[1])
        return (float(value1) + float(value2)) / 2

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

print('Dataset Length:', len(data))
# making the regressor (not being used right now)
rfr = RandomForestRegressor()

features_indices = [1, 16, 17, 19, 20, 22, 25, 26]
regressor_feature_indices = [1, 16, 17, 24, 25, 26, 27, 29]
target_indices = [0, 3, 4, 6]
regressor_target_indices = [0, 3, 4]

# get feature set for regressor
regressor_feature_set = []
for row in data:
    regressor_feature_set.append([row[i] for i in regressor_feature_indices])

regressor_target_set = []
for row in data:
    regressor_target_set.append([row[i] for i in regressor_target_indices])

# get feature set of data
feature_set = []
for row in data:
    feature_set.append([row[i] for i in features_indices])

# get target set for training set
target_set = []
for row in data:
    target_set.append([row[i] for i in target_indices])

for row in regressor_feature_set:
    row[0] = average_values(row[0])
    row[2] = average_values(row[2])

for row in regressor_target_set:
    row[0] = average_values(row[0])
    row[2] = average_values(row[2])

X_train, X_test, y_train, y_test = train_test_split(
                                    regressor_feature_set, regressor_target_set, test_size=0.2, random_state=42)


rfr.fit(X_train, y_train)
y_pred = rfr.predict(X_test)
print('TESTING REGRESSOR')
print('TEST1')
print('Actual:', y_test[10])
print('Predicted:', y_pred[10])
print()
print('TEST2')
print('Actual:', y_test[100])
print('Predicted:', y_pred[100])
print()
print('TEST3')
print('Actual:', y_test[1000])
print('Predicted:', y_pred[1000])

diffYears = []
diffPrices = []

for i in range(0, 1000):
    diffYears.append(y_pred[i][1] - float(y_test[i][1]))
    diffPrices.append(y_pred[i][0] - float(y_test[i][0]))

# Set up the figure and the two subplots
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

# Plot the first histogram
ax1.hist(diffYears, edgecolor='black')
ax1.set_title('Year - Actual vs Predicted')
ax1.set_xlabel('Year Difference')
ax1.set_ylabel('Count')

# Plot the second histogram
ax2.hist(diffPrices, edgecolor='black')
ax2.set_title('Price - Actual vs Predicted')
ax2.set_xlabel('Price Difference')
ax2.set_ylabel('Count')

plt.savefig('yearsAndPrices.png')



print('made it to classifier')
one_hot_encoder = OneHotEncoder()
feature_set = one_hot_encoder.fit_transform(feature_set)
X_train, X_test, y_train, y_test = train_test_split(
                                    feature_set, target_set, test_size=0.2, random_state=42)

print('classifying')
# Create the decision tree model
model = DecisionTreeClassifier()

# Train the model on the training data
model.fit(X_train, y_train)

# Test the model on the test data
y_pred = model.predict(X_test)
print('TESTING CLASSIFIER')
print('TEST1')
print('PREDICTED:', y_pred[1])
print('ACTUAL:', y_test[1])
print()
print('TEST2')
print('PREDICTED:', y_pred[10])
print('ACTUAL:', y_test[10])
print()
print('TEST3')
print('PREDICTED:', y_pred[100])
print('ACTUAL:', y_test[100])
print()
print('TEST4')
print('PREDICTED:', y_pred[1000])
print('ACTUAL:', y_test[1000])