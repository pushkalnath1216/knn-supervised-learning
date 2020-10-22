import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd
#import the dataset and load it into pandas dataframe
# Assign colum names to the dataset
# Read dataset to pandas dataframe
dataset = pd.read_csv('Wine.csv')  
#Preview the first five observations

dataset.head()  
dataset.shape
dataset.info()
dataset.describe()
dataset.isnull().sum()

#Preprocessing
#split the dataset into its attributes and labels.
#Stores all observations and columns into X except the last column
X = dataset.iloc[:,:-1].values  

y = dataset.iloc[:,-1].values  
print(X)
print(y)


#splits the dataset into 80% train data and 20% test data.
#out of total 150 records, the training set will contain 120 records 
#and the test set contains 30 of those records.
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9,test_size=0.10 , random_state = 0)
print(X_train)
print(X_test)
print(y_train)
print(y_test)
#Feature Scaling
#MinMaxScaler - Applies minmax normalisation
from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  

X_train.shape
X_test.shape

from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=12)  
classifier.fit(X_train, y_train)  

y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix 

cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = accuracy_score(y_test[], y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')
 
error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
    
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error')  
 

