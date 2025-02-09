import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

data_train = pd.read_csv('titanic_train.csv')
data_test = pd.read_csv('titanic_test.csv')

print("Missing values in each column:\n", data_train.isnull().sum())

data_train['Age'] = data_train['Age'].fillna(data_train['Age'].mean())

data_train['Embarked'] = data_train['Embarked'].fillna(data_train['Embarked'].mode()[0])

data_train['Fare'] = data_train['Fare'].fillna(data_train['Fare'].mean())

data_train['Cabin'] = data_train['Cabin'].fillna('Unknown')

print("\nMissing values after filling:\n", data_train.isnull().sum())

label_encoder = LabelEncoder()

data_train['Sex'] = label_encoder.fit_transform(data_train['Sex'])

data_train['Embarked'] = label_encoder.fit_transform(data_train['Embarked'])

data_train['Cabin'] = label_encoder.fit_transform(data_train['Cabin'])

features = data_train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Cabin']]

labels = data_train['Survived']

scaler = StandardScaler()

features.loc[:, ['Age', 'Fare']] = scaler.fit_transform(features[['Age', 'Fare']])

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

rf_classifier.fit(features, labels)

predictions = rf_classifier.predict(features)

accuracy = accuracy_score(labels, predictions)
print(f"Accuracy on the training set: {accuracy:.4f}")

with open('random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(rf_classifier, model_file)

with open('random_forest_model.pkl', 'rb') as model_file:
    rf_classifier_loaded = pickle.load(model_file)
data_test['Age'] = data_test['Age'].fillna(data_test['Age'].mean())
data_test['Fare'] = data_test['Fare'].fillna(data_test['Fare'].mean())
data_test['Embarked'] = data_test['Embarked'].fillna(data_test['Embarked'].mode()[0])
data_test['Cabin'] = data_test['Cabin'].fillna('Unknown')

data_test['Sex'] = label_encoder.fit_transform(data_test['Sex'])

data_test['Embarked'] = label_encoder.fit_transform(data_test['Embarked'])

data_test['Cabin'] = label_encoder.fit_transform(data_test['Cabin'])

features_test = data_test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Cabin']]

features_test.loc[:, ['Age', 'Fare']] = scaler.transform(features_test[['Age', 'Fare']])

predictions_test = rf_classifier_loaded.predict(features_test)

print("Predictions on the test data:", predictions_test)

prediction_df = pd.DataFrame({'PassengerId': data_test['PassengerId'], 'Survived': predictions_test})

prediction_df.to_csv('titanic_predictions.csv', index=False)

print("Predictions saved to 'titanic_predictions.csv'")
