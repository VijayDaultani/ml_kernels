
# coding: utf-8

# # Normalize features

# In[22]:

import numpy as np


# In[23]:

"""
np.linalg.norm : Numpy provides a function to implement the feature of 2-norm
"""
def normalize_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix, axis=0)
    normalized_features = feature_matrix / norms
    return(normalized_features, norms)


# In[24]:

"""
np.poly1d : Numpy provides a function to print the polynomials in a pretty way
"""
def print_coefficients(model):
    #Get the degree of the polynomial
    deg = len(model.coefficients['value'])-1
    
    #Get learned parameters as a list
    w = list(model.coefficients['value'])
    
    #This function for pretty printing needs the parameter in the reverse order
    print 'Learned polynomial for degree ' + str(deg) + ' : '
    w.reverse()
    print np.poly1d(w)


# In[ ]:

"""
Write wrapper for translating from string (i.e. feature name) to numpy array from data
"""
def string_numpyndarray(feature_matrix, ):
    return

