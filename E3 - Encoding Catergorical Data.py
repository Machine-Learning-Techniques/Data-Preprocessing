# Importing the necessary libraries
import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Load the dataset
dataset = pd.read_csv('titanic.csv')
#dataset.info()
#print (dataset)

# Identify the categorical data
categorical_columns = dataset.select_dtypes(include=['object']).columns.tolist()
#print (categorical_columns)

#Dropping the useless columns
dataset = dataset.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Cabin'])
#print (dataset)

#Reordering the columns to put the vector of dependent variables at the last
columns_order = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']
dataset = dataset [columns_order]
#print (dataset)

#Producing the matrix of features and the vector of dependent variables
X = dataset.iloc[: , :-1].values
y = dataset.iloc[: , -1].values

# Implement an instance of the ColumnTransformer class
ctransformer = ColumnTransformer (transformers = [('encoder', OneHotEncoder(), ([-1]))], remainder = 'passthrough')

# Apply the fit_transform method on the instance of ColumnTransformer
X = ctransformer.fit_transform(X)

# Convert the output into a NumPy array
X = np.array(X)

# Use LabelEncoder to encode binary categorical data
le = LabelEncoder()
X[:,5] = le.fit_transform(X[:,5])

# Print the updated matrix of features and the dependent variable vector
print (X)
print (y)
