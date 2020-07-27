import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import r2_score
df=pd.read_csv(r'C:\Users\chathura\Desktop\py_freq\kidney_disease.csv')
#Cleaning the DataSet
print(df.describe())
#print(df.isnull().any())
df.fillna(df.mean(),inplace=True)
df.dropna(axis=0,inplace=True)
print(df.isnull().any())
print(df.shape)
cor_df=df.corr()
sns.heatmap(cor_df)	
plt.show()
#Feature Selection
#We can see that the correlation values of Age,Id,BP, has the least signficance. So I proceed further leaving them out.
l=df.columns[3:]
X=df.iloc[:,3:25].values
y=df.iloc[:,25:26].values
#LabelEncoder for Categorical Variables
#No need for OneHotCoding , because all the categorical varibles have only two variations
lb=LabelEncoder()
for i in range(3,7):
    X[:,i]=lb.fit_transform(X[:,i])
for i in range(16,22):
    X[:,i]=lb.fit_transform(X[:,i])
y[:,0]=lb.fit_transform(y[:,0])
#Splitting the DataSet into Training and Testing parts
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=0)
#Scaling the data
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
y_train=y_train.astype('int')
#Model Evaluation
#I want to check the accuracy of 3 different models and then fix on to a model with the best accuracy
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train.ravel())
y_pred=dt.predict(X_test)
#Loading the model
dump(sc,"DT.save")
pickle.dump(dt,open('DT.pkl','wb'))
#Visualizing the classification
for i in range(list(X_test.shape)[1]):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('The '+str(l[i])+' plot')
        ax1.scatter(X_test[:,i],y_test,color='red')
        ax1.set(xlabel='The X_test values',ylabel='The y_test values')
        ax2.scatter(X_test[:,i],y_pred,color='blue')
        ax2.set(xlabel='The X_test values',ylabel='The y_pred values')
        plt.show()
    
print("The accuracy score of the Decision Tree Classifier Algorithm is ",r2_score(y_test,y_pred))
#print(y_pred)
y_pred=np.where(y_pred==1,'get Chronic Kidney Disease',y_pred)
y_pred=np.where(y_pred=='0','not get Chronic Kidney Disease',y_pred)    
print(y_pred)
