# This code is for a now closed Kaggle competition called Categorical Feature Encoding Challenge II. I achieved a score of 0.78561 out of 1 in my second try.

from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import KFold

train_data = pd.read_csv('Kaggle_categorical_train.csv')
data = train_data.drop('id',1)
label_data = data.copy()

# fixing null values
label_data['bin_0'] = label_data.bin_0.fillna(0)
label_data['bin_1'] = label_data.bin_1.fillna(0)
label_data['bin_2'] = label_data.bin_2.fillna(0)
label_data['bin_3'] = label_data.bin_3.fillna('F')
label_data['bin_4'] = label_data.bin_4.fillna('N')
label_data['nom_0'] = label_data.nom_0.fillna('Red')
label_data['nom_1'] = label_data.nom_1.fillna('Triangle')
label_data['nom_2'] = label_data.nom_2.fillna('Hamster')
label_data['nom_3'] = label_data.nom_3.fillna('India')
label_data['nom_4'] = label_data.nom_4.fillna('Theremin')
label_data['nom_5'] = label_data.nom_5.fillna('fc8fc7e56 ')
label_data['nom_6'] = label_data.nom_6.fillna('ea8c5e181')
label_data['nom_7'] = label_data.nom_7.fillna('4ae48e857')
label_data['nom_8'] = label_data.nom_8.fillna('7d7c02c57')
label_data['nom_9'] = label_data.nom_9.fillna('8f3276a6e')
label_data['ord_0'] = label_data.ord_0.fillna(1.0)
label_data['ord_1'] = label_data.ord_1.fillna('Novice')
label_data['ord_2'] = label_data.ord_2.fillna('Freezing')
label_data['ord_3'] = label_data.ord_3.fillna('n')
label_data['ord_4'] = label_data.ord_4.fillna('N')
label_data['ord_5'] = label_data.ord_5.fillna('F1')
label_data['day'] = label_data.day.fillna(3.0)
label_data['month'] = label_data.month.fillna(8.0)

# Label encoding over ordinal and nominal categorical variables
cols = label_data[['bin_3','bin_4','nom_0','nom_1','nom_2','nom_3','nom_4','nom_5','nom_6','nom_7','nom_8','nom_9','ord_0','ord_1','ord_2','ord_3','ord_4','ord_5']]

label_encoder = LabelEncoder()
for col in cols:
    label_data[col] = label_encoder.fit_transform(label_data[col])

'''''''''
This part cannot be run due to it needing a lot of memory to run

# One- Hot encoding nominal categorical variables
nominal_cols = label_data[['nom_0','nom_1','nom_2','nom_3','nom_4','nom_5','nom_6','nom_7','nom_8','nom_9']]
OH_encoder = OneHotEncoder()
array = OH_encoder.fit_transform(nominal_cols).toarray()
df = pd.DataFrame(array)

final_dataframe =pd.concat([label_data,df])
print(final_dataframe.info())

'''''''''
# Splitting the data into training and validation
predict = 'target'
X = np.array(label_data.drop([predict],1))
y = np.array(label_data[predict])

kf = KFold(n_splits= 5, shuffle= True)
for train_index, val_index in kf.split(X):
    X_train, X_val, y_train, y_val = X[train_index], X[val_index], y[train_index], y[val_index]

# Setting up test data

test_data = pd.read_csv('Kaggle_categorical_test_data.csv')
df = test_data.drop('id',1)
label_data1 = df.copy()

# fixing null values
label_data1['bin_0'] = label_data1.bin_0.fillna(0)
label_data1['bin_1'] = label_data1.bin_1.fillna(0)
label_data1['bin_2'] = label_data1.bin_2.fillna(0)
label_data1['bin_3'] = label_data1.bin_3.fillna('F')
label_data1['bin_4'] = label_data1.bin_4.fillna('N')
label_data1['nom_0'] = label_data1.nom_0.fillna('Red')
label_data1['nom_1'] = label_data1.nom_1.fillna('Triangle')
label_data1['nom_2'] = label_data1.nom_2.fillna('Hamster')
label_data1['nom_3'] = label_data1.nom_3.fillna('India')
label_data1['nom_4'] = label_data1.nom_4.fillna('Theremin')
label_data1['nom_5'] = label_data1.nom_5.fillna('e32171484')
label_data1['nom_6'] = label_data1.nom_6.fillna('4e161a54d')
label_data1['nom_7'] = label_data1.nom_7.fillna('1dddb8473')
label_data1['nom_8'] = label_data1.nom_8.fillna('d7e75499d')
label_data1['nom_9'] = label_data1.nom_9.fillna('3820773ae')
label_data1['ord_0'] = label_data1.ord_0.fillna(1.0)
label_data1['ord_1'] = label_data1.ord_1.fillna('Novice')
label_data1['ord_2'] = label_data1.ord_2.fillna('Freezing')
label_data1['ord_3'] = label_data1.ord_3.fillna('n')
label_data1['ord_4'] = label_data1.ord_4.fillna('N')
label_data1['ord_5'] = label_data1.ord_5.fillna('F1')
label_data1['day'] = label_data1.day.fillna(3.0)
label_data1['month'] = label_data1.month.fillna(8.0)

# Label encoding
columns = label_data1[['bin_3','bin_4','nom_0','nom_1','nom_2','nom_3','nom_4','nom_5','nom_6','nom_7','nom_8','nom_9','ord_0','ord_1','ord_2','ord_3','ord_4','ord_5']]

label_encoder = LabelEncoder()
for col in columns:
    label_data1[col] = label_encoder.fit_transform(label_data1[col])

X_test = np.array(label_data1)

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier()
model1 = RFC.fit(X_train,y_train)
accuracy1 = round(RFC.score(X_val,y_val)*100,2)
print('RFC:',accuracy1)


# Making the Result csv

# For RFC
if accuracy1 > 80:
    RFC_predictions = RFC.predict(X_test)

    results = {
        'target' :RFC_predictions
    }
    results_df = pd.DataFrame(results)
    id = list(range(600000,1000000,1))
    results_df['id'] = id
    results_df = results_df[['id', 'target']]
    results_df.to_csv(r'C:\Users\Mayth\OneDrive\Desktop\Categorical_Results_RFC.csv')
    print('Dataframe for RFC saved')

