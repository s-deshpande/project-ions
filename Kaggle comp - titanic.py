# This code was made for the kaggle competition called Titanic: Machine Learning from Disaster.
# This code has no data analysis but features quite a lot of data engineering.
# At the end you can see the improvements I made to the code along with the increase/decrease in accuracy.
# These improvements were made after I took several online courses.
# The highest score was achieved on my 11th try with a score of 0.78229 out of 1 and a rank of 4159.

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

data = pd.read_csv('titanic_train_data.csv')

# Fizing categorical variable Sex
gender = pd.get_dummies(data['Sex'])
data = data.join(gender)
data = data.drop(['Sex'],1)

# dropping useless variable Ticket
data = data.drop(['Ticket'], axis=1)

# checking if Name column can be used by using titles to check
for dataset in data:
    data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

for _ in data:
    data['Title'] = data['Title'].replace(['Capt', 'Col', 'Dr', 'Major', 'Rev'], 'Officer')
    data['Title'] = data['Title'].replace(['Jonkheer','Sir','Don'],'Male_Royalty')
    data['Title'] = data['Title'].replace(['Lady', 'Countess', 'Dona'], 'Female_Royalty')

    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')

title = pd.get_dummies(data['Title'])
data = data.join(title)
data = data.drop(['Title'],1)

# The passengerId and name columns can be deleted.
data = data.drop(['PassengerId', 'Name'], axis=1)

# extracting information from Cabin column
result = data.Cabin.str.extract(pat = '([A-Z])')
data['Deck'] = result
data['Deck'] = data.Deck.fillna('Unknown')

deck = pd.get_dummies(data['Deck'])
data = data.join(deck)
data = data.drop(['Deck'],1)
data = data.drop(['Cabin'], axis=1)
data = data.rename(columns = {'C':'Z'})
# changing embarked values
emb = pd.get_dummies(data['Embarked'])
data = data.join(emb)
data = data.drop(['Embarked'],1)

data['Age'] = data.Age.fillna(29) # Average age = 29
z = pd.cut(data['Age'], bins=[0,16,32,48,60,100])
data['Age'] = z
age = pd.get_dummies(data['Age'])
data = data.join(age)
data = data.drop(['Age'],1)

# Setting up training data
predict = 'Survived'
X = np.array(data.drop([predict],1))
y = np.array(data[predict])
min_max_scaler = MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)

# preventing overfitting
kf = KFold(n_splits=10,shuffle= True)
for train_index, test_index in kf.split(X):
    X_train, X_val, y_train,y_val = X[train_index],X[test_index], y[train_index], y[test_index]

# getting test data ready

test_data = pd.read_csv('titanic_test_data.csv')

gender = pd.get_dummies(test_data['Sex'])
test_data = test_data.join(gender)
test_data = test_data.drop(['Sex'],1)
# extracting information from Cabin column
result = test_data.Cabin.str.extract(pat = '([A-Z])')
test_data['Deck'] = result
test_data['Deck'] = test_data.Deck.fillna('Unknown')

test_deck = pd.get_dummies(test_data['Deck'])
test_data = test_data.join(deck)
test_data = test_data.drop(['Deck'],1)
test_data = test_data.drop(['Cabin'], axis=1)
test_data = test_data.rename(columns = {'C':'Z'})

emb = pd.get_dummies(test_data['Embarked'])
test_data = test_data.join(emb)
test_data = test_data.drop(['Embarked'],1)

test_data = test_data.drop(['Ticket'], axis=1)

for dataset in test_data:
    test_data['Title'] = test_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

for _ in test_data:
    test_data['Title'] = test_data['Title'].replace(['Capt', 'Col', 'Dr', 'Major', 'Rev'], 'Officer')
    test_data['Title'] = test_data['Title'].replace(['Jonkheer', 'Sir', 'Don'], 'Male_Royalty')
    test_data['Title'] = test_data['Title'].replace(['Lady', 'Countess','Dona'], 'Female_Royalty')

    test_data['Title'] = test_data['Title'].replace('Mlle', 'Miss')
    test_data['Title'] = test_data['Title'].replace('Ms', 'Miss')
    test_data['Title'] = test_data['Title'].replace('Mme', 'Mrs')

title = pd.get_dummies(test_data['Title'])
test_data = test_data.join(title)
test_data = test_data.drop(['Title'],1)
test_data['Male_Royalty'] = 0
test_data = test_data.drop(['PassengerId', 'Name'], axis=1)

test_data = test_data[[ 'Pclass','Age','SibSp','Parch','Fare','female','male','Female_Royalty','Male_Royalty','Master','Miss','Mr','Mrs','Officer','A','B','Z','D','E','F','G','T','Unknown','C','Q','S']]

test_data['Age'] = test_data.Age.fillna(30)
z = pd.cut(test_data['Age'], bins=[0,16,32,48,60,100])
test_data['Age'] = z
age = pd.get_dummies(test_data['Age'])
test_data = test_data.join(age)
test_data = test_data.drop(['Age'],1)

test_data['Fare'] = test_data.Fare.fillna(35.6)

X_test = np.array(test_data)
X_test_scaled= min_max_scaler.transform(X_test)

# SVM
from sklearn.svm import SVC
model2 = SVC(gamma = 0.5)
model2.fit(X_scaled,y)
predict = model2.predict(X_test_scaled)
accuracy2 = cross_val_score(model2,X,y,cv = 5)
accuracy2 = round(accuracy2.mean() *100,2)
print('SVM :',accuracy2)

# KNN
from sklearn.neighbors import KNeighborsClassifier
model3 = KNeighborsClassifier()
model3.fit(X,y)
model3.predict(X_test)
accuracy3 = cross_val_score(model3,X,y,cv = 5)
accuracy3 = round(accuracy3.mean() *100,2)
print('KNN:', accuracy3)

#  Gauusian NB
from sklearn.naive_bayes import GaussianNB
model4 = GaussianNB()
model4.fit(X,y)
model4.predict(X_test)
accuracy4 = cross_val_score(model4,X,y,cv = 5)
accuracy4 = round(accuracy4.mean() *100,2)
print('GaussianNB:', accuracy4)

# Perceptron
from sklearn.linear_model import Perceptron
model5 = Perceptron()
model5.fit(X,y)
model5.predict(X_test)
accuracy5 = cross_val_score(model5,X,y,cv = 5)
accuracy5 = round(accuracy5.mean() *100,2)
print('Perceptron:',accuracy5)

# RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
model6 = RandomForestClassifier()
model6.fit(X,y)
model6.predict(X_test)
accuracy6 = cross_val_score(model6,X,y,cv = 5)
accuracy6 = round(accuracy6.mean() *100,2)
print('RandomForestClassifier:',accuracy6)

# DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
model7 = DecisionTreeClassifier()
model7.fit(X,y)
model7.predict(X_test)
accuracy7 = cross_val_score(model7,X,y,cv = 5)
accuracy7 = round(accuracy7.mean() *100,2)
print('DecisionTreeClassifier:',accuracy7)

# run the overfitting thing as a loop over every score to select the best average ones

TARGET_PERCENT = 77

# for Random Forest

if accuracy6 > TARGET_PERCENT:
    RFC_predictions = model6.predict(X_test)

# create a results dataframe
    results = {
            "Survived": RFC_predictions
        }
    result_df = pd. DataFrame(results)
    PassengerId = list(range(892, 1310, 1))

    result_df['PassengerId'] = PassengerId
    result_df = result_df[['PassengerId', 'Survived']]
    result_df.to_csv(r'C:\Users\Mayth\OneDrive\Desktop\Titanic_Results_RFC.csv')
    print('Result Dataset for RFC saved')

# for SVM
if accuracy2 > TARGET_PERCENT:
    RFC_predictions = model2.predict(X_test_scaled)

# create a results dataframe
    results = {
            "Survived": RFC_predictions
        }
    result_df = pd. DataFrame(results)
    PassengerId = list(range(892, 1310, 1))

    result_df['PassengerId'] = PassengerId
    result_df = result_df[['PassengerId', 'Survived']]
    result_df.to_csv(r'C:\Users\Mayth\OneDrive\Desktop\Titanic_Results_SVM.csv')
    print('Result dataset for SVM saved')

# for GaussianNB
if accuracy4 > TARGET_PERCENT:
    RFC_predictions = model4.predict(X_test)

# create a results dataframe
    results = {
            "Survived": RFC_predictions
        }
    result_df = pd. DataFrame(results)
    PassengerId = list(range(892, 1310, 1))

    result_df['PassengerId'] = PassengerId
    result_df = result_df[['PassengerId', 'Survived']]
    result_df.to_csv(r'C:\Users\Mayth\OneDrive\Desktop\Titanic_Results_GNB.csv')
    print('Result dataset for GausiianNB saved')

# for KNN
if accuracy3 > TARGET_PERCENT:
    RFC_predictions = model3.predict(X_test)

# create a results dataframe
    results = {
            "Survived": RFC_predictions
        }
    result_df = pd. DataFrame(results)
    PassengerId = list(range(892, 1310, 1))

    result_df['PassengerId'] = PassengerId
    result_df = result_df[['PassengerId', 'Survived']]
    result_df.to_csv(r'C:\Users\Mayth\OneDrive\Desktop\Titanic_Results_KNN.csv')
    print('Result dataset for KNN saved')

# for DecisionTreeClassifier
if accuracy7 > TARGET_PERCENT:
    RFC_predictions = model7.predict(X_test)

# create a results dataframe
    results = {
            "Survived": RFC_predictions
        }
    result_df = pd. DataFrame(results)
    PassengerId = list(range(892, 1310, 1))

    result_df['PassengerId'] = PassengerId
    result_df = result_df[['PassengerId', 'Survived']]
    result_df.to_csv(r'C:\Users\Mayth\OneDrive\Desktop\Titanic_Results_DTC.csv')
    print('Result dataset for DTC saved')


# Score of 70.07 % , can try to improve it using feature engineering - First try
# Score of 72.72 %, using around 86% accuracy from RFC after overfitting fix - Second try
# Score of 76.56 %, using around 91% accuracy from KNN- third try
# Score of 77.99 % with 86% accuracy from SVM after Deck addition to dataframe. However saw a mistake with the TITLE section in test_data code
# Score of 78.947 & with 89.89% accuracy on SVM after multiple things like Title were fixed
# Score of  80.382 with 91.01% accuracy on SVM with pd.get_dummies


# I suspect some overfitting still present.May have to perform more feature engineering
# or better models(logistic regression)
# Like use stratified KFold
