# predicts car to buy based on car sold to carmax

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
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

print(data)
print(len(data))
print(len(data[0]))
print(nulls)