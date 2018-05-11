# Random Forest Classification

# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
X_train = dataset.iloc[:, [0,2, 4,5,6,7]].values
y_train = dataset.iloc[:, 1].values


test_dataset = pd.read_csv('test.csv')
X_test= test_dataset.iloc[:, [0,1,3,4,5,6]].values


# taking care of missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_train[:, 3:4])
X_train[:, 3:4] = imputer.transform(X_train[:, 3:4])
imputer_test = imputer.fit(X_test[:, 3:4])
X_test[:, 3:4] = imputer_test.transform(X_test[:, 3:4])


# One hot encoder
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelEncoder_X = LabelEncoder()
X_train[:, 2] = labelEncoder_X.fit_transform(X_train[:, 2])
oneHotEncoder= OneHotEncoder(categorical_features = [2])
X_train = oneHotEncoder.fit_transform(X_train).toarray()

X_test[:, 2] = labelEncoder_X.fit_transform(X_test[:, 2])
oneHotEncoder= OneHotEncoder(categorical_features = [2])
X_test = oneHotEncoder.fit_transform(X_test).toarray()

# Splitting the dataset into the Training set and Test set
'''from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)'''

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

