#Breast Cancer Prediction
#Importing the required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Reading the data
data=pd.read_csv('data_2.csv')
#Checking if missing data is present or not
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()  

#Converting the categorical variable to dummy variable(0 or 1)
diag=pd.get_dummies(data['diagnosis'],drop_first=True)
data.drop(['diagnosis','id'],axis=1,inplace=True)
data=pd.concat([data,diag],axis=1)


X=data.iloc[: ,0:30]
y=data.iloc[:, 31:32]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Using Gaussian Naive Bayes model to train the model
from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
clf.fit(X_train,y_train)
#Predicting the result
predict=clf.predict(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix
a=accuracy_score(predict,y_test)  #0.9190*100=91.9%
cm=confusion_matrix(y_test,predict)
'''[[84,  6],
       [ 6, 47]], dtype=int64)'''
