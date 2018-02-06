import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#importing keras libraries and modules
import keras
from keras.models import Sequential   #used to initialize ann
from keras.layers import Dense		  #used to create layers of ann

#initializing the ANN
classifier = Sequential()

#adding input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu", input_dim = 11))  # units = number of nodes in hidden layers 
														   # kernel_initializer = initializing weights close to 0
														   # activation = activation function for hidden layer 'relu' is for rectifier fynction 
														   # input_dim = no of input layers

#adding second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))	

#adding output layer
classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))	#sigmoid for output layer as result is binary

#compiling the ANN

classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"]) #optimizer = function to use to optimize the weights in our case it is stochastic gradient descent 
										 							   #different types of SGD available.we are using adam
										 							   #loss = cost function used to optimize weights of neural network.here we are using logistic regression cost function

#fitting the ANN to training set
classifier.fit(X_train,y_train,batch_size = 10,nb_epoch = 100)

#predicting the test set results

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#making the confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)