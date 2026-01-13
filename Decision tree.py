
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
import sklearn.metrics as metrics


import matplotlib.pyplot as plt
import seaborn as sns


# Data loading and inspection
df_original = pd.read_csv("Invistico_Airline.csv")

df_original.head(10)
df_original.dtypes
df_original["Class"].unique()
df_original["satisfaction"].value_counts(dropna=False)
df_original.isnull().sum()
df_original.shape


# Data cleaning
df_subset = df_original.dropna(axis=0).reset_index(drop=True)

df_subset.isna().sum()
df_subset.shape

# Encoding
df_subset["Class"] = df_subset["Class"].map({
    "Business": 3,
    "Eco Plus": 2,
    "Eco": 1
})

df_subset["satisfaction"] = df_subset["satisfaction"].map({
    "satisfied": 1,
    "dissatisfied": 0
})

df_subset = pd.get_dummies(df_subset, drop_first=True)
df_subset.dtypes

# Feature / target split
y = df_subset["satisfaction"]

X = df_subset.drop("satisfaction", axis=1)

# 
# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# Baseline Decision Tree
decision_tree = DecisionTreeClassifier(random_state=0)
decision_tree.fit(X_train, y_train)

dt_pred = decision_tree.predict(X_test)

print("Decision Tree")
print("Accuracy:", "%.6f" % metrics.accuracy_score(y_test, dt_pred))
print("Precision:", "%.6f" % metrics.precision_score(y_test, dt_pred))
print("Recall:", "%.6f" % metrics.recall_score(y_test, dt_pred))
print("F1 Score:", "%.6f" % metrics.f1_score(y_test, dt_pred))


# Confusion Matrix
cm = metrics.confusion_matrix(
    y_test, dt_pred, labels=decision_tree.classes_
)

disp = metrics.ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=decision_tree.classes_
)

disp.plot()
plt.show()
# Baseline tree visualization
plt.figure(figsize=(20, 12))
plot_tree(
    decision_tree,
    max_depth=2,
    fontsize=14,
    feature_names=X.columns
)

# Baseline feature importance
importances = decision_tree.feature_importances_
forest_importances = (
    pd.Series(importances, index=X.columns)
    .sort_values(ascending=False)
)

fig, ax = plt.subplots()
forest_importances.plot.bar(ax=ax)
plt.show()

# Hyperparameter grid
tree_para = {
    "max_depth": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40,50],
    "min_samples_leaf": [2,3,4,5,6,7,8,9,10,15,20,50]
}
# Scoring 
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}


# Tuned Decision Tree
tuned_decision_tree = DecisionTreeClassifier(random_state=0)

clf = GridSearchCV(
    tuned_decision_tree,
    tree_para,
    scoring=scoring,
    cv=5,
    refit="f1"
)

clf.fit(X_train, y_train)

clf.best_estimator_

print("Best Avg. Validation Score:", "%.4f" % clf.best_score_)

# Results table function
results = pd.DataFrame(
    columns=["Model", "F1", "Recall", "Precision", "Accuracy"]
)

def make_results(model_name, model_object):
    """
    Returns a dataframe with F1, Recall, Precision, and Accuracy
    for the model with the highest mean F1 across CV folds.
    """

    cv_results = pd.DataFrame(model_object.cv_results_)

    best_estimator_results = cv_results.iloc[
        cv_results["mean_test_f1"].idxmax(), :
    ]

    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy

    table = pd.DataFrame({
        "Model": [model_name],
        "F1": [f1],
        "Recall": [recall],
        "Precision": [precision],
        "Accuracy": [accuracy]
    })

    return table

result_table = make_results("Tuned Decision Tree", clf)
result_table

# Tuned tree visualization
plt.figure(figsize=(20, 12))
plot_tree(
    clf.best_estimator_,
    max_depth=2,
    fontsize=14,
    feature_names=X.columns
)
plt.figure(figsize=(20, 12))
plot_tree(decision_tree, max_depth=2, fontsize=14, feature_names=X.columns)
plt.show()

# Tuned feature importance 
importances = clf.best_estimator_.feature_importances_

forest_importances = (
    pd.Series(importances, index=X.columns)
    .sort_values(ascending=False)
)

fig, ax = plt.subplots()
forest_importances.plot.bar(ax=ax)
plt.show()
