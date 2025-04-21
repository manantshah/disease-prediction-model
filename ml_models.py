#############################
# Project 1 - Problem 2     #
# Author : Manan Tarun Shah #
# Python version 3.9.7      #
#############################

import pandas as pd                                     # to read data
import numpy as np                                      # for array and math functions
from sklearn.model_selection import train_test_split    # to split database
from sklearn.tree import DecisionTreeClassifier         # for Decision Tree ML method
from sklearn.metrics import accuracy_score              # to fetch accuracy results
from sklearn.linear_model import Perceptron             # for Perceptron ML method
from sklearn.preprocessing import StandardScaler        # standardize data
from sklearn.svm import SVC                             # for Support Vector Machine ML method
from sklearn.ensemble import RandomForestClassifier     # for Random Forest ML method
from sklearn.neighbors import KNeighborsClassifier      # for KNN ML method
from sklearn.linear_model import LogisticRegression     # for Logistic Regression ML method
from tabulate import tabulate                           # for output in table format

df = pd.read_csv('heart1.csv')                          # read in the data

# list of all variable/feature names
column_list = ['age', 'sex', 'cpt', 'rbp', 'sc', 'fbs', 'rer', 'mhr', 'eia', 'opst', 'dests', 'nmvcf', 'thal', 'a1p2']
X = df[column_list[:13]]    # our data includes all the variables/features except a1p2
y = df['a1p2']              # our target or the variable which we want to predict is a1p2

# split the problem into train and test
# this will yield 70% training and 30% test
# random_state allows the split to be reproduced
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()                  # creating standard scalar
sc.fit(X_train)                        # computing required transformation
X_train_std = sc.transform(X_train)    # applying to training data
X_test_std = sc.transform(X_test)      # and SAME transformation of test data

# perceptron linear
# epoch is one forward and backward pass of all training samples
# (also known as an iteration)
# eta0 is rate of convergence
# max_iter, tol, if it is too low it is never achieved
# and continues to iterate to max_iter when above tol
# fit_intercept, fit the intercept or assume it is 0
# slowing it down is very effective, eta is the learning rate
ppn = Perceptron(max_iter=6, tol=1e-3, eta0=0.001, fit_intercept=True, random_state=0, verbose=False)
ppn.fit(X_train_std, y_train)    # training

# print('Number in test ', len(y_test))
y_pred_ppn = ppn.predict(X_test_std)    # try with the test data

# Note that this only counts the samples where the predicted value was wrong
# print('Misclassified samples: %d' % (y_test != y_pred_ppn).sum())
# idea about performance, how'd we do?
print('Perceptron ML method:')
print('Test Accuracy: %.2f' % accuracy_score(y_test, y_pred_ppn))

# vstack puts first array above the second in a vertical stack
# hstack puts first array to left of the second in a horizontal stack
# NOTE the double parentheses!
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
# print('Number in combined ', len(y_combined))

# did the stack so we can see how the combination of test and train data did
y_combined_pred_ppn = ppn.predict(X_combined_std)
# print('Misclassified combined samples: %d' % (y_combined != y_combined_pred_ppn).sum())
comb_acc_ppn = accuracy_score(y_combined, y_combined_pred_ppn)
print('Combined Accuracy: %.2f\n' % comb_acc_ppn)

# Support Vector Machine (kernel-linear)
# kernel - specify the kernel type to use
# C - the penalty parameter - it controls the desired margin size. Larger C, larger penalty
svm = SVC(kernel='linear', C=0.7, random_state=0)
svm.fit(X_train_std, y_train)    # training

y_pred_svm = svm.predict(X_test_std)   # work on test data

# Results
# print('Number in test ', len(y_test))
# print('Misclassified samples: %d' % (y_test != y_pred_svm).sum())
print('Support Vector Machine (kernel-linear) ML method:')
print('Test Accuracy: %.2f' % accuracy_score(y_test, y_pred_svm))

# combine the train and test sets
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# and analyze the combined sets
# print('Number in combined ', len(y_combined))
y_combined_pred_svm = svm.predict(X_combined_std)
# print('Misclassified combined samples: %d' % (y_combined != y_combined_pred_svm).sum())
comb_acc_svm = accuracy_score(y_combined, y_combined_pred_svm)
print('Combined Accuracy: %.2f\n' % comb_acc_svm)

# create the Decision Tree classifier and train it
tree = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=0)
tree.fit(X_train, y_train)    # training

y_pred_tree = tree.predict(X_test)    # try with the test data
# print('Number in test ', len(y_test))
# print('Misclassified samples: %d' % (y_test != y_pred_tree).sum())
print('Decision Tree Learning ML method:')
print('Test Accuracy: %.2f' % accuracy_score(y_test, y_pred_tree))

# combine the train and test data
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

# see how we do on the combined data
y_combined_pred_tree = tree.predict(X_combined)
# print('Misclassified combined samples: %d' % (y_combined != y_combined_pred_tree).sum())
comb_acc_tree = accuracy_score(y_combined, y_combined_pred_tree)
print('Combined Accuracy: %.2f\n' % comb_acc_tree)

# create the Random Forest classifier and train it
# n_estimators is the number of trees in the forest
# the entropy choice grades based on information gained
# n_jobs allows multiple processors to be used
# print("Number of trees: ", trees)
forest = RandomForestClassifier(criterion='entropy', n_estimators=11, random_state=1, n_jobs=4)
forest.fit(X_train, y_train)    # training

y_pred_for = forest.predict(X_test)    # see how we do on the test data
# print('Number in test ', len(y_test))
# print('Misclassified samples: %d' % (y_test != y_pred_for).sum())
print('Random Forest ML method:')
print('Test Accuracy: %.2f' % accuracy_score(y_test, y_pred_for))

# combine the train and test data
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

# see how we do on the combined data
y_combined_pred_for = forest.predict(X_combined)
# print('Misclassified samples: %d' % (y_combined != y_combined_pred_for).sum())
comb_acc_for = accuracy_score(y_combined, y_combined_pred_for)
print('Combined Accuracy: %.2f\n' % comb_acc_for)

# create the K-Nearest Neighbors classifier and fit it
# since only 2 features, minkowski is same as euclidean distance
# where p=2 specifies sqrt(sum of squares). (p=1 is Manhattan distance)
# print(neighs, 'neighbors')
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)    # training

# run on the test data and print results and check accuracy
y_pred_knn = knn.predict(X_test_std)
# print('Number in test ', len(y_test))
# print('Misclassified samples: %d' % (y_test != y_pred_knn).sum())
print('K-Nearest Neighbors ML method:')
print('Test Accuracy: %.2f' % accuracy_score(y_test, y_pred_knn))

# combine the train and test data
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# check results on combined data
y_combined_pred_knn = knn.predict(X_combined_std)
# print('Misclassified samples: %d' % (y_combined != y_combined_pred_knn).sum())
comb_acc_knn = accuracy_score(y_combined, y_combined_pred_knn)
print('Combined Accuracy: %.2f\n' % comb_acc_knn)

# Create logistic regression component.
# C is the inverse of the regularization strength. Smaller -> stronger!
# C is used to penalize extreme parameter weights.
# solver is the particular algorithm to use
# multi_class determines how loss is computed
# ovr -> binary problem for each label
lr = LogisticRegression(C=1, solver='liblinear', multi_class='ovr', random_state=0)
lr.fit(X_train_std, y_train)    # apply the algorithm to training data

# run on the test data and print results and check accuracy
y_pred_lr = lr.predict(X_test_std)
# print('Number in test ', len(y_test))
# print('Misclassified samples: %d' % (y_test != y_pred_lr).sum())
print('Logistic Regression ML method:')
print('Test Accuracy: %.2f' % accuracy_score(y_test, y_pred_lr))

# combine the train and test data
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# check results on combined data
# print('Number in combined ', len(y_combined))
y_combined_pred_lr = lr.predict(X_combined_std)
# print('Misclassified combined samples: %d' % (y_combined != y_combined_pred_lr).sum())
comb_acc_lr = accuracy_score(y_combined, y_combined_pred_lr)
print('Combined Accuracy: %.2f\n' % comb_acc_lr)

# printing combined accuracy for all ML methods in table format
print("OUTPUT:")
comb_acc_output = [[1, 'Perceptron', round(100 * comb_acc_ppn)],
                   [2, 'Support Vector Machine (kernel-linear)', round(100 * comb_acc_svm)],
                   [3, 'Decision Tree Learning', round(100 * comb_acc_tree) ],
                   [4, 'Random Forest', round(100 * comb_acc_for)],
                   [5, 'K-Nearest Neighbors', round(100 * comb_acc_knn)],
                   [6, 'Logistic Regression', round(100 * comb_acc_lr)]]
print(tabulate(comb_acc_output, headers=["No.", "Machine Learning method", "Combined accuracy(%)"]))