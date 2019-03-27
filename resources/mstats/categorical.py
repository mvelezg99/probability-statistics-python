# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 11:28:58 2019

@author: Miguel Ángel Vélez
"""

import numpy as np
#------------------------------------------------------------------------------------------------#
#-------------------------------- Categorical Data Analysis -------------------------------------#
#------------------------------------------------------------------------------------------------#

def get_x2(oi, ei):
    """
    Function to calculate the statistical test for the categorical data analysis,
    it is a chi squared value.
    
    Parameters:
    --------------------------
    oi : list
        Frequency from the observed events.
    ei : list
        Frequency from the expected events.
    
    Returns:
    --------------------------
    x2 : double
        Chi squared value which represents the calculated statistical test.
    """
    if (any(isinstance(el, list) for el in oi)): # Validate if the list is a list of lists
        x2 = 0
    
        for i in range(len(oi)):
            for j in range(len(oi[i])):
                x2 += (oi[i][j] - ei[i][j])**2 / ei[i][j]
            
        return x2
    else:
        k = len(oi)
        x2 = sum([((oi[x] - ei[x])**2 / ei[x]) for x in range(k)])
        return x2
        
    

def get_ei(oi, pi):
    """
    """
    n = sum(oi)
    ei = [(n * pi[x]) for x in range(len(oi))]
    
    return ei

def get_pi(distr, intervals, oi):
    """
    """
    pi = [(distr.cdf(x[1]) - distr.cdf(x[0])) for x in intervals]
    return pi

def contingency_ei(oi):
    """
    """
    n = sum(sum(x) for x in oi)
    rows = list(map(list, zip(*oi)))
    tot_rows = np.array([sum(x) for x in rows])
    tot_cols = np.array([sum(x) for x in oi])
    
    pi_rows = np.array([(tot_rows[x] / n) for x in range(len(tot_rows))])
    
    ei = (tot_cols[..., None] * pi_rows[None, ...]).tolist()
    # np.outer(tot_cols, pi_rows) -> It is the same numpy form to make the
    # outer product of two vectors.
    
    return ei
    
    
            

    

#------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------#
