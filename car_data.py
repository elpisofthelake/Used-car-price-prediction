# Data Preprocessing

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('D:\\Berojgari\\ML_projects\\car_data.csv')
print(dataset.head())

# Data exploration
print(dataset.shape)
print(dataset.columns)
print(dataset.info())

# Categorical columns
categorical_cols = dataset.select_dtypes(include='object').columns
print(categorical_cols)
print(len(categorical_cols))

# Numerical columns
numerical_cols = dataset.select_dtypes(include=['float64', 'int64']).columns
print(numerical_cols)
print(len(numerical_cols))

print(dataset.describe())

# Dealing with missing values
print(dataset.isnull().values.any())
print(dataset.isnull().sum())

# Restructure the dataset
print(dataset.head())

dataset = dataset.drop(columns='name')
print(dataset.head())

# Adding a column
dataset['Current Year'] = 2024
dataset['Years Old'] = dataset['Current Year'] - dataset['year']
print(dataset.head())

dataset = dataset.drop(columns=['Current Year', 'year'])
print(dataset.head())

# Encoding the categorical data
print(dataset.select_dtypes(include='object').columns)
print(len(dataset.select_dtypes(include='object').columns))
print(dataset['fuel'].nunique())
print(dataset['seller_type'].nunique())
print(dataset['transmission'].nunique())
print(dataset['owner'].nunique())
print(dataset.shape)

# One-hot encoding
dataset = pd.get_dummies(data=dataset, drop_first=True)
print(dataset.head())
print(dataset.shape)

# Correlation matrix
dataset_2 = dataset.drop(columns='selling_price')
dataset_2.corrwith(dataset['selling_price']).plot.bar(
    figsize=(16, 9), title='Correlated with Selling Price', grid=True
)

# Applying log transformation
dataset['selling_price_log'] = np.log1p(dataset['selling_price'])

# Re-run the correlation matrix to see if there are better correlated features
corr = dataset.corr()
plt.figure(figsize=(16, 9))
sns.heatmap(corr, annot=True)

# Splitting the dataset

# Matrix of features
x = dataset.drop(columns=['selling_price', 'selling_price_log'])

# Target variable
y = dataset['selling_price_log']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Building the neural network model

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Initializing the neural network
model = Sequential()

# Adding layers
model.add(Dense(units=64, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1))

# Compiling the neural network
model.compile(optimizer='adam', loss='mse')

# Training the neural network
model.fit(x_train, y_train, epochs=100, batch_size=10)

# Predicting the test set results
y_pred = model.predict(x_test)

# Evaluating the model
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))

# Predicting a single observation
print(dataset.head())
single_obs = np.array([[70000, 17, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1]])
print(model.predict(single_obs))

