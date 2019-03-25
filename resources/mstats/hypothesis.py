# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:40:23 2019

@author: Miguel Angel Velez
"""

import scipy.stats as ss

#------------------------------------------------------------------------------------------------#
#------------------------------------- Hypothesis Testing ---------------------------------------#
#------------------------------------------------------------------------------------------------#

def crit_val_norm(sign, tail):
    """
    Function to calculate, with the standard normal distribution, the critical 
    values according to the level of significance and the tail-type test.
    
    Parameters:
    --------------------------
    sign : float, double
        Level of significance of the test.
        Each value should be in the range [0, 1].
        
    tail : string
        Tail-type test, it could be 'left', 'right' or 'two'.
        
    Returns:
    --------------------------
    crit_val : tuple, float
        The critical values or value, according to the level of significance and
        the tail-type test.
    """
    if tail == 'two':
        alpha = 1 - (sign/2)
        crit_val = ss.norm.ppf(1 - alpha), ss.norm.ppf(alpha)
        return crit_val
    
    if tail == 'left':
        alpha = 1 - sign
        crit_val = ss.norm.ppf(1 - alpha)
        return crit_val
    
    if tail == 'right':
        alpha = 1 - sign
        crit_val = ss.norm.ppf(alpha)
        return crit_val
    
    print("You must input a valid tail ('two', 'left' or 'right')")
    

def crit_val_t(df, sign, tail):
    """
    Function to calculate, with the t-student distribution, the critical 
    values according to the level of significance and the tail-type test.
    
    Parameters:
    --------------------------
    df : int
        Degrees of freedom (n - 1). It's the sample size minus 1.
        
    sign : float, double
        Level of significance of the test.
        Each value should be in the range [0, 1].
        
    tail : string
        Tail-type test, it could be 'left', 'right' or 'two'.
        
    Returns:
    --------------------------
    crit_val : tuple, float
        The critical values or value, according to the level of significance and
        the tail-type test.
    """
    if tail == 'two':
        alpha = 1 - (sign/2)
        crit_val = ss.t.ppf(1 - alpha, df), ss.t.ppf(alpha, df)
        return crit_val
    
    if tail == 'left':
        alpha = 1 - sign
        crit_val = ss.t.ppf(1 - alpha, df)
        return crit_val
    
    if tail == 'right':
        alpha = 1 - sign
        crit_val = ss.t.ppf(alpha, df)
        return crit_val
    
    print("You must input a valid tail ('two', 'left' or 'right')")
    

def crit_val_f(df1, df2, sign):
    """
    """
    alpha = 1 - sign
    crit_val = ss.f.ppf(alpha, df1, df2)
    return crit_val

def crit_val_chi2(df, sign):
    """
    """
    alpha = 1 - sign
    crit_val = ss.chi2.ppf(alpha, df)
    return crit_val
            
    
def reject_h0(crit_val, value, tail):
    """
    Function to determine if reject the null hypothesis or not reject it based
    on the tail-type test.
    
    Parameters:
    --------------------------
    crit_val : tuple, float
        Critical values to consider.
        
    value : float, double
        Value to compare with critical values.
        
    tail : string
        Tail-type test, it could be 'left', 'right' or 'two'.
        
    Returns:
    --------------------------
    decision : bool
        True if we should reject null hypothesis,
        False if we shouldn't reject null hypothesis.
    """
    if tail == 'two':
        return (value < crit_val[0] or value > crit_val[1])
    
    if tail == 'left':
        return value < crit_val
    
    if tail == 'right':
        return value > crit_val
    
    print("You must input a valid tail ('two', 'left' or 'right')")
    
    
def get_p(z, tail):
    """
    Function to determine the p-value (minimum significance level to reject
    the null hypothesis).
    
    Parameters:
    --------------------------
    z : double, float
        Z-value of the test statistic.
        
    tail : string
        Tail-type test, it could be 'left', 'right' or 'two'.
        
    Returns:
    --------------------------
    p : double, float
        P-value, the minimum significance level to reject the null hypothesis.
    """
    z_area = ss.norm.cdf(abs(z))
    
    if tail == 'two':
        p = 2 * (1 - z_area)
        return p
    
    else:
        p = 1 - z_area
        return p

            
    
#------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------#