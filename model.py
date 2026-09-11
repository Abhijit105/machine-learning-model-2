import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, root_mean_squared_error, mean_absolute_error, f1_score, confusion_matrix

#########################################################################################

df = pd.read_csv("./student_performance_extended_ml_ready.csv")
print("\n##### 5 sample records: #####")
print(df.sample(5))

print("\n##### Data shape: #####")
print(df.shape)

print("\n##### Data size: #####")
print(df.size)

print("\n##### Data column names: #####")
print(df.columns)

#########################################################################################

X = df.drop(['passed'], axis=1)
y = df.iloc[:, -1]
print("\n##### X sample: #####")
print(X.sample(5))

print("\n##### X shape: #####")
print(X.shape)

print("\n##### y sample: #####")
print(y.sample(5))

print("\n##### y shape: #####")
print(y.shape)

#########################################################################################

model_logistic_regression = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print("\n##### X_train shape: #####")
print(X_train.shape)

print("\n##### X_test shape: #####")
print(X_test.shape)

print("\n##### y_train shape: #####")
print(y_train.shape)

print("\n##### y_test shape: #####")
print(y_test.shape)
model_logistic_regression.fit(X_train, y_train)
y_pred = model_logistic_regression.predict(X_test)

accuracy_logistic_regression = accuracy_score(y_test, y_pred)
f1_logistic_regression = f1_score(y_test, y_pred)
mae_logistic_regression = mean_absolute_error(y_test, y_pred)
rmse_logistic_regression = root_mean_squared_error(y_test, y_pred)
confusion_matrix_logistic_regression = confusion_matrix(y_test, y_pred)

print("\n##### Accuracy logistic regression: #####")
print(accuracy_logistic_regression)

print("\n##### F1 logistic regression: #####")
print(f1_logistic_regression)

print("\n##### MAE logistic regression: #####")
print(mae_logistic_regression)

print("\n##### RMSE logistic regression: #####")
print(rmse_logistic_regression)

print("\n##### Confusion matrix logistic regression: #####")
print(confusion_matrix_logistic_regression)


#########################################################################################

model_decision_tree = DecisionTreeClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print("\n##### X_train shape: #####")
print(X_train.shape)

print("\n##### X_test shape: #####")
print(X_test.shape)

print("\n##### y_train shape: #####")
print(y_train.shape)

print("\n##### y_test shape: #####")
print(y_test.shape)
model_decision_tree.fit(X_train, y_train)
y_pred = model_decision_tree.predict(X_test)

accuracy_decision_tree = accuracy_score(y_test, y_pred)
f1_decision_tree = f1_score(y_test, y_pred)
mae_decision_tree = mean_absolute_error(y_test, y_pred)
rmse_decision_tree = root_mean_squared_error(y_test, y_pred)
confusion_matrix_decision_tree = confusion_matrix(y_test, y_pred)

print("\n##### Accuracy decision tree: #####")
print(accuracy_decision_tree)

print("\n##### F1 decision tree: #####")
print(f1_decision_tree)

print("\n##### MAE decision tree: #####")
print(mae_decision_tree)

print("\n##### RMSE decision tree: #####")
print(rmse_decision_tree)

print("\n##### Confusion matrix decision tree: #####")
print(confusion_matrix_decision_tree)