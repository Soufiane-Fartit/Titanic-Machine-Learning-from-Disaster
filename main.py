import numpy as np
import pandas as pd

from sklearn import preprocessing, linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix

df = pd.read_csv("train.csv")
df = df.fillna("missing")
df.head()

features = df[['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']]
labels = df[['Survived']]

features.head()

#MAKE PREDICTIONS
testing_data = pd.read_csv("test.csv")
testing_data = testing_data.fillna("missing")

testing_data = testing_data[['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']]

merged = pd.concat([df,testing_data])

uniques_sex = pd.unique(merged[['Sex']].values.ravel('K'))
uniques_cabin = pd.unique(merged[['Cabin']].values.ravel('K'))
uniques_embarked = pd.unique(merged[['Embarked']].values.ravel('K'))

le_sex = preprocessing.LabelEncoder()
le_cabin = preprocessing.LabelEncoder()
le_embarked = preprocessing.LabelEncoder()

le_sex.fit(uniques_sex)
le_cabin.fit(uniques_cabin)
le_embarked.fit(uniques_embarked)

list(le_sex.classes_)
list(le_cabin.classes_)
list(le_embarked.classes_)

#print(features['Age'].mean())
features['Age'] = features['Age'].replace(["missing"], 23.8)

features['Sex'] = features['Sex'].replace(list(le_sex.classes_), le_sex.transform(list(le_sex.classes_)))
features['Cabin'] = features['Cabin'].replace(list(le_cabin.classes_), le_cabin.transform(list(le_cabin.classes_)))
features['Embarked'] = features['Embarked'].replace(list(le_embarked.classes_), le_embarked.transform(list(le_embarked.classes_)))

features.head()
labels.head()

#classifier = KNeighborsClassifier()
classifier = MLPClassifier()
#classifier = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
classifier.fit(features,labels)
predicted_labels = classifier.predict(features)
print(confusion_matrix(labels, predicted_labels))
print(classification_report(labels, predicted_labels))


#MAKE PREDICTIONS
testing_data = pd.read_csv("test.csv")
testing_data = testing_data.fillna("missing")


testing_data = testing_data[['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']]

testing_data['Age'] = testing_data['Age'].replace(["missing"], 23.8)
testing_data['Fare'] = testing_data['Fare'].replace(["missing"], 32)

testing_data['Sex'] = testing_data['Sex'].replace(list(le_sex.classes_), le_sex.transform(list(le_sex.classes_)))
testing_data['Cabin'] = testing_data['Cabin'].replace(list(le_cabin.classes_), le_cabin.transform(list(le_cabin.classes_)))
testing_data['Embarked'] = testing_data['Embarked'].replace(list(le_embarked.classes_), le_embarked.transform(list(le_embarked.classes_)))

features.head()
testing_data.head()
testing_predicted_labels = classifier.predict(testing_data)
print(len(testing_predicted_labels))

dd = pd.DataFrame({'Survived': testing_predicted_labels }, index=[892+i for i in range(418)])
print(dd.head())
dd.to_csv('predicted.csv')
