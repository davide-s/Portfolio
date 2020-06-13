import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn

from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

address = 'train.csv'
titanic_training = pd.read_csv(address)
titanic_training.columns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
print(titanic_training.head())

print(titanic_training.info())

#Checking for missing values

print(titanic_training.isnull().sum())

#Dropping missing values

titanic_data = titanic_training.drop(['Name', 'Ticket','Cabin'], axis=1)

print(titanic_data.head())

#Imputing missing values

sb.boxplot(x='Parch', y= 'Age', data=titanic_data, palette= "hls")
plt.show()

parch_groups =  titanic_data.groupby(titanic_data['Parch'])
print(parch_groups.mean())

def age_approx(cols):
    age = cols[0]
    parch = cols[1]

    if pd.isnull(age):
        if parch == 0:
            return 32
        elif parch == 1:
            return 24
        elif parch == 2:
            return 17
        elif parch == 3:
            return 33
        elif parch == 4:
            return 45
        else: return 30
           
    else: 
        return age 

titanic_data['Age'] = titanic_data[['Age','Parch']].apply(age_approx, axis=1)

print(titanic_data.isnull().sum())

titanic_data.dropna(inplace = True)
titanic_data.reset_index(inplace=True, drop=True)

print(titanic_data.info())

#Converting categorical variables to a dummy indicators

label_encoder = LabelEncoder()
gender_cat = titanic_data['Sex']
gender_encoded = label_encoder.fit_transform(gender_cat)
print(gender_encoded[0:5])

print(titanic_data.head())

#1 = male 0 = female

gender_DF = pd.DataFrame(gender_encoded, columns=['male_gender'])

embarked_cat = titanic_data['Embarked']
embarked_encoded = label_encoder.fit_transform(embarked_cat)
embarked_encoded[0:100]

binary_encoder = OneHotEncoder(categories='auto')
embarked_1hot = binary_encoder.fit_transform(embarked_encoded.reshape(-1,1))
embarked_1hot_mat = embarked_1hot.toarray()
embarked_DF = pd.DataFrame(embarked_1hot_mat, columns=['C','Q','S'])
print(embarked_DF.head())

titanic_data.drop(['Sex', 'Embarked'], axis=1, inplace=True)
titanic_data.head()

titanic_dmy = pd.concat([titanic_data, gender_DF, embarked_DF], axis=1, verify_integrity=True).astype(float)
titanic_dmy[0:5]

#Checking for independence between features

sb.heatmap(titanic_dmy.corr())
titanic_dmy.drop(['Fare', 'Pclass'], axis=1, inplace=True)
titanic_dmy.head()