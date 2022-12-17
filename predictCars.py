# predicts car to buy based on car sold to carmax

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
import csv

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

print('Dataset Length:', len(data))
# making the regressor
rfr = RandomForestRegressor()

features_indices = [16, 17, 19, 20, 22, 25, 26]
target_indices = [0, 3, 4, 6]

# get feature set of data
feature_set = []
for row in data:
    feature_set.append([row[i] for i in features_indices])

# get target set for training set
target_set = []
for row in data:
    target_set.append([row[i] for i in target_indices])

one_hot_encoder = OneHotEncoder()
feature_set = one_hot_encoder.fit_transform(feature_set)
X_train, X_test, y_train, y_test = train_test_split(
                                    feature_set, target_set, test_size=0.2, random_state=42)


# Create the decision tree model
model = DecisionTreeClassifier()

# Train the model on the training data
model.fit(X_train, y_train)

# Test the model on the test data
y_pred = model.predict(X_test)
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


# Calculate the accuracy of the model
# accuracy = model.score(X_test, y_test)
# print("Accuracy: ", accuracy)