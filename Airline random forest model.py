
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, PredefinedSplit, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import pickle


air_data = pd.read_csv("Invistico_Airline.csv")
air_data.head(10)
air_data.dtypes
air_data.shape
air_data.isna().any(axis=1).sum()

air_data_subset = air_data.dropna(axis=0)

air_data_subset.head(10)
air_data_subset.isna().sum()



air_data_subset_dummies = pd.get_dummies(air_data_subset, columns=['Customer Type', 'Type of Travel', 'Class'])
air_data_subset_dummies.head()
air_data_subset_dummies.head(10)
air_data_subset_dummies.dtypes


y = air_data_subset_dummies['satisfaction']
X = air_data_subset_dummies.drop('satisfaction', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)


cv_params = {'n_estimators': [25, 50, 100],
             'max_depth': [None, 10, 25, 50],
             'min_samples_split': [2,5, 10],
             'min_samples_leaf': [1, 2, 3],
             'max_features': ['sqrt'],
             'max_samples': [None, 0.5, 0.8]}
        
        
split_index = [0 if x in X_val.index else -1 for x in X_train.index]
custom_split = PredefinedSplit(split_index)

# Instantiate Model
rf = RandomForestClassifier(random_state=0)


# Search over specified parameters.
rf_val = GridSearchCV(rf, cv_params, cv=custom_split, refit='f1', n_jobs = -1, verbose = 1)

# Fit your model.
rf_val.fit(X_train, y_train)

# Find optimal parameters.
rf_val.best_params_

# Use optimal parameters on GridSearchCV.
rf_opt = RandomForestClassifier(n_estimators = 50, max_depth = 50, 
                                min_samples_leaf = 1, min_samples_split = 0.001,
                                max_features="sqrt", max_samples = 0.9, random_state = 0)

# Fit the optimal model.
rf_opt.fit(X_train, y_train)

# Predict on test set.
y_pred = rf_opt.predict(X_test)

# Get precision score.
pc_test = precision_score(y_test, y_pred, pos_label = "satisfied")
print("The precision score is {pc:.3f}".format(pc = pc_test))

# Get recall score.
rc_test = recall_score(y_test, y_pred, pos_label = "satisfied")
print("The recall score is {rc:.3f}".format(rc = rc_test))

# Get accuracy score.
ac_test = accuracy_score(y_test, y_pred)
print("The accuracy score is {ac:.3f}".format(ac = ac_test))

# Get F1 score.
f1_test = f1_score(y_test, y_pred, pos_label = "satisfied")
print("The F1 score is {f1:.3f}".format(f1 = f1_test))

### Evaluate the model

# Precision score on test data set.
print("\nThe precision score is: {pc:.3f}".format(pc = pc_test), "for the test set,", "\nwhich means of all positive predictions,", "{pc_pct:.1f}% prediction are true positive.".format(pc_pct = pc_test * 100))

# Recall score on test data set.
print("\nThe recall score is: {rc:.3f}".format(rc = rc_test), "for the test set,", "\nwhich means of which means of all real positive cases in test set,", "{rc_pct:.1f}% are  predicted positive.".format(rc_pct = rc_test * 100))

# Accuracy score on test data set.
print("\nThe accuracy score is: {ac:.3f}".format(ac = ac_test), "for the test set,", "\nwhich means of all cases in test set,", "{ac_pct:.1f}% are predicted true positive or true negative.".format(ac_pct = ac_test * 100))

# F1 score on test data set.
print("\nThe F1 score is: {f1:.3f}".format(f1 = f1_test), "for the test set,", "\nwhich means the test set's harmonic mean is {f1_pct:.1f}%.".format(f1_pct = f1_test * 100))

# Table of results.
table = pd.DataFrame({'Model': ["Tuned Decision Tree", "Tuned Random Forest"],
                        'F1':  [0.945422, f1_test],
                        'Recall': [0.935863, rc_test],
                        'Precision': [0.955197, pc_test],
                        'Accuracy': [0.940864, ac_test]
                      }
                    )
table

