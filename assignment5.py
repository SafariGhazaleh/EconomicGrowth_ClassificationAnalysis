#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ghazaleh Safari 

Integrated International Master- and PhD program in Mathematics 
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

"""Load the datasets"""
filename_OECD_data = "assignment_5_OECD_data_explaining_vars.csv"
filename_OECD_data_growth = "assignment_5_OECD_data_growth.csv"
df = pd.read_csv(filename_OECD_data, header=0)
df_eci = pd.read_csv(filename_OECD_data_growth, header=0)

"""Step 1: Data Preprocessing and merging the target variable
   Merge the datasets based on 'REG_ID' as the key"""
merged_df = pd.merge(df, df_eci, left_on='REG_ID', right_on='REG_ID', how='inner')

""" Step 2: Handle non-numeric values
    Identify the columns with non-numeric values"""
non_numeric_columns = merged_df.select_dtypes(include=['object']).columns

"""Perform one-hot encoding on the non-numeric columns"""
merged_df = pd.get_dummies(merged_df, columns=non_numeric_columns)

"""Step 3: Split the data into features and target variable"""
X = merged_df.drop(columns=['Growth > 5%']).values
y = merged_df['Growth > 5%'].values

"""Step 4: Train the models"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

"""Model 1: Decision Tree Classifier"""
dt_classifier = DecisionTreeClassifier(random_state=0)
dt_classifier.fit(X_train, y_train)

"""Model 2: Random Forest Classifier"""
rf_classifier = RandomForestClassifier(random_state=0)
rf_classifier.fit(X_train, y_train)

"""Model 3: Multilayer Perceptron Classifier"""
mlp_classifier = MLPClassifier(random_state=0)
mlp_classifier.fit(X_train, y_train)

"""Step 5: Evaluate the models"""
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, f1, confusion

dt_accuracy, dt_precision, dt_recall, dt_f1, dt_confusion = evaluate_model(dt_classifier, X_test, y_test)
rf_accuracy, rf_precision, rf_recall, rf_f1, rf_confusion = evaluate_model(rf_classifier, X_test, y_test)
mlp_accuracy, mlp_precision, mlp_recall, mlp_f1, mlp_confusion = evaluate_model(mlp_classifier, X_test, y_test)

"""Step 6: Interpret the results"""
print("Decision Tree Classifier:")
print("Accuracy:", dt_accuracy)
print("Precision:", dt_precision)
print("Recall:", dt_recall)
print("F1 Score:", dt_f1)
print("Confusion Matrix:\n", dt_confusion)

print("\nRandom Forest Classifier:")
print("Accuracy:", rf_accuracy)
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1 Score:", rf_f1)
print("Confusion Matrix:\n", rf_confusion)

print("\nMultilayer Perceptron Classifier:")
print("Accuracy:", mlp_accuracy)
print("Precision:", mlp_precision)
print("Recall:", mlp_recall)
print("F1 Score:", mlp_f1)
print("Confusion Matrix:\n", mlp_confusion)
