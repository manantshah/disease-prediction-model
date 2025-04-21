############################
# Project 1 - Problem 1    #
# Author: Manan Tarun Shah #
# Python version 3.9.7     #
############################

import pandas as pd                # for reading data
import numpy as np                 # for arrays and math
import matplotlib.pyplot as plt    # for plotting purpose
import seaborn as sns              # data visualization

df = pd.read_csv('heart1.csv')     # read in the data

# CORRELATION CALCULATION STARTS #

# calculating correlation and taking absolute values as large negatives are as useful as large positives
corr = df.corr().abs()
# print(corr)

# set the correlations on the diagonal or lower triangle to zero, so they will not be reported as the highest ones.
# (The diagonal is always 1; the matrix is symmetric about the diagonal.)

# We clear the diagonal since the correlation with itself is always 1.

# Note the * in front of the argument in tri. That's because shape returns
# a tuple and * unrolls it so they become separate arguments.

# Note this will be element by element multiplication
corr *= np.tri(*corr.values.shape, k=-1).T
# print(corr)

# now unstack it so we can sort things
# note that zeros indicate no correlation OR that we cleared below the diagonal
corr_unstack = corr.unstack()
# print(corr_unstack)

# Sort values in descending order
# To determine correlation of each variable with all the other variables with high correlation values at top
corr_unstack.sort_values(inplace=True, ascending=False)
print("Correlation of each variable with all other variables in descending order:\n", corr_unstack)

# Now just print the top values to determine which variables are most highly correlated with each other
print("\nTop 5 most highly correlated variables with each other:\n", corr_unstack.head(5))

# Get the correlations with a1p2 which is the variable we wish to predict
with_a1p2_corr = corr_unstack.get(key="a1p2")
# print(with_a1p2_corr)

# Now just print the top values to determine which variables are most highly correlated with a1p2
print("\nTop 6 most highly correlated variables with a1p2(the variable we wish to predict):\n", with_a1p2_corr.head(6))

# COVARIANCE CALCULATION STARTS #

# calculating covariance and taking absolute values as large negatives are as useful as large positives
cov = df.cov().abs()    # creating cross covariance matrix
# print(cov)

# set the covariance on the diagonal or lower triangle to zero, so they will not be reported as the highest ones.
# (The diagonal is always 1; the matrix is symmetric about the diagonal.)

# We clear the diagonal since the covariance with itself is always 1.

# Note the * in front of the argument in tri. That's because shape returns
# a tuple and * unrolls it so they become separate arguments.

# Note this will be element by element multiplication
cov *= np.tri(*cov.values.shape, k=-1).T
# print(cov)

# now unstack it so we can sort things
# note that zeros indicate no covariance OR that we cleared below the diagonal
cov_unstack = cov.unstack()
# print(cov_unstack)

# Sort values in descending order
# Shows variables that are highly dependent of each other at the top
cov_unstack.sort_values(inplace=True, ascending=False)
print("\nCovariance of each variable with all other variables in descending order:\n", cov_unstack)

# Printing the top values to indicate variables which show highest covariance
print("\nTop 5 values that show variables with highest covariance:\n", cov_unstack.head(5))

# Get the covariance with a1p2 which is the variable we wish to predict
with_a1p2_cov = cov_unstack.get(key="a1p2")
# print(with_a1p2_cov)

# Now just print the top values to determine which variables have highest covariance with a1p2
# These are the best predictors of the heart disease
print("\nTop 5 values showing variables having highest covariance with a1p2:\n", with_a1p2_cov.head(5))

sns.set(style='darkgrid', context='notebook', font_scale=0.7)     # set appearance
sns.pairplot(df, height=1.5)                                      # create pair plot
plt.show()                                                        # display it (might take about 30 seconds to load)