
# coding: utf-8

# In[11]:

import numpy as np


# In[12]:

"""
Perform the prediction on the data
"""
def predict_output(feature_matrix,weights):
    predictions = feature_matrix.dot(weights)
    return predictions


# In[13]:

"""
Single Coordinate Descent Step
"""
def lasso_coordiante_descent_step(i, feature_matrix, output, weights, l1_penalty):
    #Remove below after checking if the output's type
    output = np.reshape(output, len(output))
    
    #Compute prediction
    prediction = predict_output(feature_matrix,weights)
    
    #Compute ro[i] = SUM[ [feature_i]*(output - prediction + weight[i]*[feature_i]) ]
    ro_i = (feature_matrix[:,i] * (output - prediction + weights[i] * feature_matrix[:,i])).sum()

    if i == 0: # intercept -- do not regularize
        new_weight_i = ro_i 
    elif ro_i < -l1_penalty/2.:
        new_weight_i = ro_i + (l1_penalty/2)
    elif ro_i > l1_penalty/2.:
        new_weight_i = ro_i - (l1_penalty/2)
    else:
        new_weight_i = 0.
    
    return new_weight_i


# In[14]:

"""
Cyclical Coordinate Descent Step
"""
def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):
    change = np.zeros(len(initial_weights))
    change.fill(tolerance+1)
    #print change
    while(np.amax(change) > tolerance):
        for i in range(len(initial_weights)):
            #Remember the old weights
            old_weights_i = initial_weights[i]
            #Calculate the lasso coordinate step
            initial_weights[i] = lasso_coordinate_descent_step(i, feature_matrix, output, initial_weights, l1_penalty)
            #Calculate change in ith weight
            #print initial_weights
            #print("Old Weight for feature ") + str(i) + (" is :") + str(old_weights_i) + (" New weight is : ") + str(initial_weights[i])
            change[i] = abs(initial_weights[i] - old_weights_i)
        #print("Tolerance is :") + str(tolerance) + (" Maximum change is :") + str(np.amax(change))
    return initial_weights


# In[ ]:



