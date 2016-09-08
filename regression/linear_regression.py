
# coding: utf-8

# # Following will be the steps we will follow to perform linear regression on the dataset.
# 
# * Perform ridge regression.
# * Perform lasso regression.
# * Perform NN regression.
# 

# # Step 1 : Import Necessary Libraries

# In[ ]:

"""
First of all we will import necessary packages required for our tasks of exploration,
visualization, analysis and fitting models and measuring how good they are
"""
#Import numpy and pandas
import numpy as np
import pandas as pd

#Import seaborn and matplotlib for plotting
import seaborn as sns
import matplotlib.pyplot as plt
#Below line is used to make sure matplotlib plots all graphs on the same notebook
get_ipython().magic(u'matplotlib inline')

#Current version of seaborn issues some warnings, we shall ignore them for now
import warnings
warnings.filterwarnings("ignore")

#Import solver
from preprocessing_data import normalize_features
reload(preprocessing_data)

#from coordinate_descent import lasso_cyclical_coordinate_descent
import coordinate_descent


# # Step 2 : Load data 

# In[ ]:

"""
Load data in abalone_df Pandas DataFrame
We use delimeter= " " because in the dataset the columns are separated using " "
Also we specify names of the column explicitly to ease our understanding.
"""
abalone_df = pd.read_csv("../../data/abalone/Dataset.data", delimiter=" ",
                         names=['Sex', 'Length', 'Diameter', 'Height', 
                                'Whole weight', 'Shucked weight', 'Viscera weight',
                                'Shell weight', 'Rings'])


# # Step 3 : Start exploring dataset 

# In[93]:

#First of all lets look how big the dataset is using len function
print "Number of Observations in the dataset : " + str(len(abalone_df))

#Now lets have a glimpse of first few rows of the dataset
abalone_df.head()


# # Step 4 : Perform some preprocessing
# * Normalize all features.
# * Perform feature engineering to get some interaction and different features.
# * Look for High leverage and Influential observations in the code.
# * Divide data into training, validation and test set

# In[94]:

"""
First of all normalize all the features present in the dataset. 
We use function as_matrix of pandas to create the numpy into its numpy array representation
"""
abalone_df_normalized, norms = normalize_features(abalone_df.as_matrix(columns=['Length','Diameter','Height', 
                                'Whole weight', 'Shucked weight', 'Viscera weight',
                                'Shell weight', 'Rings']))


# # Regression Method 1 : Least square regression
# * Explain the objective function of Least square regression

# # Regression Method 2 : Perform ridge regression (a.k.a regularization) 
# * Explain the obective function of ridge regression.
# * Perform ridge regression
# * Evaluate l2 penalties using cross validation and choose the best one. Draw graph for distribution
# * Evaluate on test data

# # Regression Method 3 : Perform lasso regression 
# * Explaining the objective function of lasso
# * Implementing single coordinate descent step
# * Implementing Cyclical Coordinate descent algorithm
# * Evaluate l1 penalties using cross validation and choose the best one. Draw graph for distribution
# * Evaluate on test data
# * Implement debiasing (lasso followed by least square (not regularization)

# In[95]:

weights = coordinate_descent.lasso_cyclical_coordinate_descent(normalized_simple_feature_matrix, output,
                                            initial_weights, l1_penalty, tolerance)


# # Regression Method 4 : Perform NN regression
# * 1D NN
# * Distance metric : Euclidean and Manhattan

# In[10]:




# In[ ]:



