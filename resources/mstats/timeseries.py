# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:40:27 2019

@author: Miguel Ángel Vélez
"""

from mstats import linregr

#------------------------------------------------------------------------------------------------#
#------------------------------------------- Time Series ----------------------------------------#
#------------------------------------------------------------------------------------------------#

#x = [52, 81, 47, 65, 50, 73, 45, 60, 50, 79, 45, 62]
#
#x2 = [40, 45, 38, 47, 53, 39, 47, 32, 51, 45, 37, 54]
#
#x3 = [70, 68, 75, 79, 67, 81, 82, 69, 72, 68]
#
#x4 = [105, 110, 107, 112, 117, 109, 108]

def get_rm(x, N):
    """
    """
    cumsum, rm = [0], []
    
    for i, j in enumerate(x, 1):
        cumsum.append(cumsum[i - 1] + j)
        
        if i >= N:
            mean = (cumsum[i] - cumsum[i - N]) / N
            rm.append(mean)
            
    return rm

def running_mean(x, N):
    """
    """
    rm = get_rm(x, N)
    
    if(N%2 == 0):
        return get_rm(rm, 2)
    else:
        return rm
        

#print(running_mean(x3, 3))
        

def next_F(At, Ft, alpha):
    """
    """
    next_F = alpha * At + (1 - alpha) * Ft
    return next_F

#print(next_F(110, 105, 0.3))

def get_error(At, Ft):
    """
    """
    return Ft - At

def get_projections(x, alpha):
    """
    """
    projections = []
    
    for i in range(len(x) + 1):
        if(i == 0):
            projections.append("-")
        elif(i == 1):
            projections.append(x[i - 1])
        else:
            projection = next_F(x[i - 1], projections[i - 1], alpha)
            projections.append(projection)
    
    return projections

def get_errors(x, alpha):
    """
    """
    projections = get_projections(x, alpha)
    errors = []
    
    for i in range(1, len(x)):
        error = get_error(x[i], projections[i])
        errors.append(error)
        
    return errors
        
def get_MSE(x, alpha):
    """
    """
    errors = get_errors(x, alpha)
    
    MSE = sum([(errors[i])**2 for i in range(len(errors))]) / (len(errors))
    
    return MSE
    
def trendline(y, t):
    """
    """
    model = linregr.regression(t, y)
    return model
    
    
#print(get_projections(x4, 0.3))
#print(get_errors(x4, 0.3))
#print(get_MSE(x4, 0.3))



#------------------------------------------------------------------------------------------------#

