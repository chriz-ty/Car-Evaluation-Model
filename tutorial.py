import  sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from  sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")   #reading the data from the csv file

le = preprocessing.LabelEncoder()   #using preprocessing module for encoding categorical labels as neumerical values

#transforming categorical columns in the dataset into neumerical values

buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"  # will be used later as the target variable you want to predict.

#combines the encoded categorical features into a single feature matrix x using the zip function
x = list(zip(buying,maint,door,persons,lug_boot,safety))

y = list(cls)   #creates a list of target values

#splits the dataset into training and testing sets
x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=9)  # creates a k-nearest neighbors classifier with 9 neighbors

model.fit(x_train,y_train)    # trains the k-nearest neighbors classifier on the training data
acc=model.score(x_test,y_test)  #calculates the accuracy of the model on the testing data
print(acc)

#this part makes predictions on the test data using the trained KNN classifier

predicted = model.predict(x_test)  #uses the trained KNN model (model) to make predictions on the test data (x_test).
names = ["unacc", "acc", "good", "vgood"]
for i in range(len(predicted)):
    print("Predicted: ", names[predicted[i]], "Data: ",x_test[i], "Actual: ",names[y_test[i]] )
