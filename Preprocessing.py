#Data Preprocessing

#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the Dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

#Taking care of missing data
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer() #Missing values replaced by mean of other values
# imputer.fit(X[:, 1:3])
# X[:, 1:3] = imputer.transform(X[:, 1:3])

#Encoding Categorical data
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
# X = ct.fit_transform(X)
# labelencoder_X = LabelEncoder()
#X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
#X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
#X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
#for x in range(3):
    #X[:, x] = labelencoder_X.fit_transform(X[:, x])
#onehotencoder = OneHotEncoder(categorical_features = [0])
#X = onehotencoder.fit_transform(X).toarray()

# labelencoder_Y = LabelEncoder()
# Y = labelencoder_Y.fit_transform(Y) 

#Splitting the Dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)