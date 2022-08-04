# -*- coding: utf-8 -*-
"""Kaggle_Data_contest
"""

"""
Please before running the code specify the paths below in the "paths for the Dataset" section 
and then the noteboook can be Run completely genrating the Final CSV.
"""

"""
Mounting Drive
"""

from google.colab import drive
drive.mount('/content/drive')

"""Imports"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.datasets import make_hastie_10_2
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import matthews_corrcoef

"""Paths for the Datasets"""

"""Dataset 1 paths"""
train_path1="/content/drive/MyDrive/Data_contest/Dataset_1/Dataset_1_Training.csv"
test_path1="/content/drive/MyDrive/Data_contest/Dataset_1/Dataset_1_Testing.csv"

"""Dataset 2 paths"""
train_path2="/content/drive/MyDrive/Data_contest/Dataset_2/Dataset_2_Training.csv"
test_path2="/content/drive/MyDrive/Data_contest/Dataset_2/Dataset_2_Testing.csv"

"""Path to save the generated CSV file"""

save_path='/content/drive/MyDrive/Data_contest/CSV/lr_lr_lr_lr_20ada_20ada.csv'

"""Preprocessing using scalling"""

def data_preprocessing_1(train_path,test_path):
  train = pd.read_csv(train_path)
  test = pd.read_csv(test_path)
  df = train.T
  test_df = test.T
  train = df[1:]
  test = test_df[1:]
  Actual_train = train.iloc[: , :-2]
  C1 = train.iloc[: , -2:-1]
  C1 = np.array(C1.values.astype(int))
  C2 = train.iloc[: , -1:]
  C2 = np.array(C2.values.astype(int))
  scaler = StandardScaler()
  scaler.fit(Actual_train)
  Actual_train = scaler.transform(Actual_train)
  scaler.fit(test)
  test = scaler.transform(test)
  return Actual_train,test,C1,C2

X_train1,test1,C1,C2=data_preprocessing_1(train_path1,test_path1)

def data_preprocessing_2(train_path,test_path):
  train = pd.read_csv(train_path)
  test = pd.read_csv(test_path)
  df = train.T
  test_df = test.T
  train = df[1:]
  test = test_df[1:]
  Actual_train = train.iloc[: , :-4]
  C3 = train.iloc[: , -4:-3]
  C3 = np.array(C3.values.astype(int))
  C4 = train.iloc[: , -3:-2]
  C4 = np.array(C4.values.astype(int))
  C5 = train.iloc[: , -2:-1]
  C5 = np.array(C5.values.astype(int))
  C6 = train.iloc[: , -1:]
  C6 = np.array(C6.values.astype(int))
  scaler = StandardScaler()
  scaler.fit(Actual_train)
  Actual_train = scaler.transform(Actual_train)
  scaler.fit(test)
  test = scaler.transform(test)
  return Actual_train,test,C3,C4,C5,C6

X_train2,test2,C3,C4,C5,C6=data_preprocessing_2(train_path2,test_path2)

"""**PCA for Dimension Reduction**"""

'''
from sklearn.decomposition import PCA
pca=PCA()
X_train1=pca.fit_transform(X_train1)
X_train1.shape
'''

'''
from sklearn.decomposition import PCA
pca=PCA()
X_train2=pca.fit_transform(X_train2)
X_train2.shape
'''

"""**DataSet_1**

C0:1 Descriptor
"""

model1 = LogisticRegression(max_iter=500)
model1.fit(X_train1 , C1)
print(model1.predict(test1))

"""C0:2 Descriptor"""

model2 = LogisticRegression(max_iter=500)
model2.fit(X_train1 , C2)
print(model2.predict(test1))

"""**DataSet 2**

Hyperparameters for AdaBoost Algorithm
"""

n_estimators=20
learning_rate=0.46

"""Hyperparameter Tuning for SVM and Adaboost"""

'''
param_svm = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

param_ada = {'base_estimator__max_depth':[i for i in range(2,11,2)],
              'base_estimator__min_samples_leaf':[5,10],
              'n_estimators':[10,20,50,250],
              'learning_rate':[0.01,0.1,0.4,0.5,0.6,0.456]}

model = GridSearchCV(model, parameters,verbose=3,scoring='f1',n_jobs=-1)
'''

"""C0:3 Descriptor"""

'''
from imblearn.over_sampling import SMOTE
smt=SMOTE()
X_sm,y_sm = smt.fit_resample(X_train2, C3)'''
model3 = LogisticRegression(max_iter=500)
model3.fit(X_train2 , C3)
model3.predict(test2)

"""C0:4 Descriptor"""

'''
from imblearn.over_sampling import SMOTE
smt=SMOTE()
X_sm,y_sm = smt.fit_resample(X_train2, C4)'''
model4 = LogisticRegression(max_iter=500)
model4.fit(X_train2 ,C4)
model4.predict(test2)

"""C0:5 Descriptor"""

model5 = AdaBoostClassifier(n_estimators=n_estimators,learning_rate=learning_rate)
model5.fit(X_train2 , C5)
model5.predict(test2)

"""C0:6 Descriptor

"""

model6 = AdaBoostClassifier(n_estimators=n_estimators,learning_rate=learning_rate)
model6.fit(X_train2 , C6)
model6.predict(test2)

"""Generating the Resultant predictions"""

def csv_generate(test1,test2,model1,model2,model3,model4,model5,model6):
  result_csv=[]
  Temp = model1.predict(test1)
  for i  in Temp:
      result_csv.append(i)
  Temp = model2.predict(test1)
  for i  in Temp:
      result_csv.append(i)  
  Temp = model3.predict(test2)
  for i  in Temp:
      result_csv.append(i)
  Temp = model4.predict(test2)
  for i  in Temp:
      result_csv.append(i)  
  Temp = model5.predict(test2)
  for i  in Temp:
      result_csv.append(i)
  Temp = model6.predict(test2)
  for i  in Temp:
      result_csv.append(i)
  return result_csv

"""Generating the CSV"""

result_csv=csv_generate(test1,test2,model1,model2,model3,model4,model5,model6)
len(result_csv)

"""Storing the CSV"""

def csv_generate(result_csv,save_path):
  id=[]
  for i in range(len(result_csv)):
    id.append(i)
  mydf = pd.DataFrame(list(zip(id, result_csv)), columns = ['Id', 'Predicted'])
  mydf.to_csv(save_path,index=False)

csv_generate(result_csv,save_path)

"""**Model Comparison**

Define X, y for the corressponding descriptor to compare individually.
"""

'''
X=X_train2
y=C5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
'''

"""Define the corresponding Model for the Descriptor"""

'''
model=GaussianNB()
model.fit(X_train,y_train)
'''

#y_pred=model.predict(X_test)

"""Prediction Score using Kaggle mentioned MCC"""

#print(matthews_corrcoef(y_test,y_pred))

"""**Data Visualization**

SVD for visualizing on only high variance feature vectors.
"""

'''
from sklearn.decomposition import TruncatedSVD

svd =  TruncatedSVD(n_components = 2)
df = svd.fit_transform(X_train2)

#print("Transformed Matrix after reducing to 2 features:")
#print(df)
'''

'''
X1=[]
X2=[]
y=[]
for i in df:
  X1.append(i[0])
  X2.append(i[1])
c=[]
for i in C4:
  y.append(i)
  if i==0:
    c.append("Class 0")
  else:
    c.append('Class 1')
'''

'''
sns.scatterplot(X1,X2,hue=c)
plt.legend()
plt.savefig("/content/drive/MyDrive/Data_contest/plots/C4_pair.png")
plt.show()
'''

'''
from sklearn.decomposition import TruncatedSVD
svd =  TruncatedSVD(n_components = 1)
df = svd.fit_transform(X_train2)

#print("Transformed Matrix after reducing to 1 feature:")
#print(df)
'''

'''
X=[]
y=[]
for i in df:
  X.append(i[0])
c=[]
for i in C4:
  y.append(i)
  if i==0:
    c.append("Class 0")
  else:
    c.append('Class 1')
'''

'''
sns.scatterplot(X,y,hue=c)
plt.legend()
plt.savefig("/content/drive/MyDrive/Data_contest/plots/C4_highestVariance.png")
plt.show()
'''

"""**Correlation Visulaization**"""

'''
from sklearn.decomposition import TruncatedSVD
svd =  TruncatedSVD(n_components = 4)
df = svd.fit_transform(X_train2)
df=pd.DataFrame(df)
'''

"""After SVD"""

'''
f, ax = plt.subplots(figsize=(10, 6))
corr = df.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.3f',
                 linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('Correlation Heatmap After SVD', fontsize=14)
plt.savefig('/content/drive/MyDrive/Data_contest/plots/Datset1correlation_after.png')
plt.show()
'''

"""Before SVD"""

'''
df_original=pd.DataFrame(X_train2).iloc[:,:4]
f, ax = plt.subplots(figsize=(10, 6))
corr = df_original.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.3f',
                 linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('Correlation Heatmap before SVD', fontsize=14)
plt.savefig('/content/drive/MyDrive/Data_contest/plots/Datset1correlation_before.png')
plt.show()
'''

"""**Class Imbalancing**"""

'''
from imblearn.over_sampling import SMOTE
import collections
'''

'''
y=[]
for i in C4:
  y.append(i[0])
counter=collections.Counter(y)
counter
'''

#smt=SMOTE()

#X_sm,y_sm = smt.fit_resample(X_train2, C4)

#print(collections.Counter(y_sm))

'''
X1=[]
X2=[]
y=[]
for i in X_sm:
  X1.append(i[0])
  X2.append(i[1])
c=[]
for i in y_sm:
  y.append(i)
  if i==0:
    c.append("Class 0")
  else:
    c.append('Class 1')
'''

'''
sns.scatterplot(X1,X2,hue=c)
plt.savefig('/content/drive/MyDrive/Data_contest/plots/afterSMOTE_C4.png')
plt.show()
'''

