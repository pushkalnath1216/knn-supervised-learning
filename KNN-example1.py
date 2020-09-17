import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_excel('D:\course docs\ML\datasets\m and f.xlsx')
data.head()
data.describe()

x=data[['ht','wt']]
y=data['sex']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=0)

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
scaler.fit(x_train)

x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

x_train.shape
x_test.shape

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=12)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix,accuracy_score

cm=confusion_matrix(y_pred,y_test)

accuracy=accuracy_score(y_test,y_pred)*100
print('accuracy of the model is'+str(round(accuracy,2))+'%')

error=[]
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred_i=knn.predict(x_test)
    error.append(np.mean(pred_i!=y_test))

plt.figure(figsize=(12,6))
plt.plot(range(1,40),error,color='red',linestyle='dashed',marker='o',markerfacecolor='blue',markersize=10)
plt.title('error rate-K value')
plt.xlabel('k value')
plt.ylabel('mean error')    
    