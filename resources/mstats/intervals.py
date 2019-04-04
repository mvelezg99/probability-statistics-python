# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:36:13 2019

@author: Miguel Angel Velez
"""

import scipy.stats as ss
from mstats import generals, linregr

#------------------------------------------------------------------------------------------------#
#------------------------------------- Confidence Intervals -------------------------------------#
#------------------------------------------------------------------------------------------------#

def norm(n, x_, sd, alpha):
    """
    Function to get the confidence interval to estimate the average population
    when we have large samples (n >= 30).
    
    Parameters:
    --------------------------
    n : int
        Sample size.
        
    x_ : float, double
        Sample average.
    
    sd : float, double
        Sample or population standard deviation.
        
    alpha : float, double
        Confidence level to estimate our unknown parameter (the average population).
        Each value should be in the range [0, 1].
    
    Returns:
    --------------------------
    interval : tuple
        The confidence interval, with the minimum value at [0] and the maximum value
        at [1] for the unknown parameter (the average population).
    """
    if (n <= 30):
        print("The sample size must be greater than 30.")
    else:
        sd_error = sd / (n**0.5)
        z_value = ss.norm.interval(alpha)[1]
        
        minimum = x_ - (z_value * sd_error)
        maximum = x_ + (z_value * sd_error)
        
        return minimum, maximum
    
def norm_2p(n1, n2, x_1, x_2, sd1, sd2, alpha):
    """
    Function to get the confidence interval to estimate the difference between
    2 population averages in large samples (n >= 30).
    
    Parameters:
    --------------------------
    n1 : int
        Sample 1 size.
        
    n2 : int
        Sample 2 size.
        
    x_1 : float, double
        Sample 1 average.
    
    x_2 : float, double
        Sample 2 average.
    
    sd1 : float, double
        Sample or population 1 standard deviation.
        
    sd2 : float, double
        Sample or population 2 standard deviation.
        
    alpha : float, double
        Confidence level to estimate our unknown parameter (the difference between
        2 population averages).
        Each value should be in the range [0, 1].
    
    Returns:
    --------------------------
    interval : tuple
        The confidence interval, with the minimum value at [0] and the maximum value
        at [1] for the unknown parameter (the difference between
        2 population averages).
    """
    if (n1 <= 30 or n2 <= 30):
        print("The sample sizes must be greater than 30.")
    else:
        sd_error = ((sd1**2 / n1) + (sd2**2 / n2))**0.5
        z_value = ss.norm.interval(alpha)[1]
        
        minimum = (x_1 - x_2) - (z_value * sd_error)
        maximum = (x_1 - x_2) + (z_value * sd_error)
        
        return minimum, maximum

           
def t(n, x_, s, alpha):
    """
    Function to get the confidence interval to estimate the average population
    when we have small samples (n < 30).
    
    Parameters:
    --------------------------
    n : int
        Sample size.
        
    x_ : float, double
        Sample average.
    
    s : float, double
        Sample standard deviation.
        
    alpha : float, double
        Confidence level to estimate our unknown parameter (the average population).
        Each value should be in the range [0, 1].
    
    Returns:
    --------------------------
    interval : tuple
        The confidence interval, with the minimum value at [0] and the maximum value
        at [1] for the unknown parameter (the average population).
    """    
    if (n > 30):
        print("The sample size must be less than 30.")
    else:
        df = n - 1
        s_error = s / (n**0.5)
        t_value = ss.t.interval(alpha, df)[1]
        
        minimum = x_ - (t_value * s_error)
        maximum = x_ + (t_value * s_error)
        
        return minimum, maximum

def t_2p(n1, n2, x_1, x_2, s1, s2, alpha, var):
    """
    Function to get the confidence interval to estimate the difference between
    2 population averages in small samples (n < 30).
    
    Parameters:
    --------------------------
    n1 : int
        Sample 1 size.
        
    n2 : int
        Sample 2 size.
        
    x_1 : float, double
        Sample 1 average.
    
    x_2 : float, double
        Sample 2 average.
    
    s1 : float, double
        Sample 1 standard deviation.
        
    s2 : float, double
        Sample 2 standard deviation.
        
    alpha : float, double
        Confidence level to estimate our unknown parameter (the difference between
        2 population averages).
        Each value should be in the range [0, 1].
        
    var : bool
        Are the population variances equal?
        True if population variances are equal.
        False if population variances are different.
    
    Returns:
    --------------------------
    interval : tuple
        The confidence interval, with the minimum value at [0] and the maximum value
        at [1] for the unknown parameter (the difference between
        2 population averages).
    """    
    if (n1 > 30 or n2 > 30):
        print("The sample sizes must be less than 30.")
    else:
        if (var is True):
            df = n1 + n2 - 2
            s2p = (((s1**2) * (n1 - 1)) + (s2**2 * (n2 - 1))) / df
            sp_error = ((s2p / n1) + (s2p / n2))**0.5
            t_value = ss.t.interval(alpha, df)[1]
            
            minimum = (x_1 - x_2) - (t_value * sp_error)
            maximum = (x_1 - x_2) + (t_value * sp_error)
        
            return minimum, maximum
        else:
            df = generals.get_df_var(n1, n2, s1, s2)
            
            sp_error = (((s1**2) / n1) + ((s2**2) / n2))**0.5
            t_value = ss.t.interval(alpha, df)[1]
            
            minimum = (x_1 - x_2) - (t_value * sp_error)
            maximum = (x_1 - x_2) + (t_value * sp_error)
            
            return minimum, maximum


def prop(n, p, alpha):
    """
    Function to get the confidence interval to estimate the population proportion.
    
    Parameters:
    --------------------------
    n : int
        Sample size.
        
    p : float, double
        Sample proportion.
        Each value should be in the range [0, 1].
            
    alpha : float, double
        Confidence level to estimate our unknown parameter (the population proportion).
        Each value should be in the range [0, 1].
    
    Returns:
    --------------------------
    interval : tuple
        The confidence interval, with the minimum value at [0] and the maximum value
        at [1] for the unknown parameter (the population proportion).
    """   
    sp_error = ((p * (1 - p)) / n)**0.5
    z_value = ss.norm.interval(alpha)[1]
    
    minimum = p - (z_value * sp_error)
    maximum = p + (z_value * sp_error)
    
    return minimum, maximum

def prop_2p(n1, n2, p1, p2, alpha):
    """
    Function to get the confidence interval to estimate the difference between
    2 population proportions.
    
    Parameters:
    --------------------------
    n1 : int
        Sample 1 size.
    
    n2 : int
        Sample 2 size.
        
    p1 : float, double
        Sample 1 proportion.
        Each value should be in the range [0, 1].
        
    p2 : float, double
        Sample 2 proportion.
        Each value should be in the range [0, 1].
            
    alpha : float, double
        Confidence level to estimate our unknown parameter (the population proportion).
        Each value should be in the range [0, 1].
    
    Returns:
    --------------------------
    interval : tuple
        The confidence interval, with the minimum value at [0] and the maximum value
        at [1] for the unknown parameter (the difference between
        2 population proportions).
    """
    sp_error = (((p1 * (1 - p1)) / n1) + ((p2 * (1 - p2)) / n2))**0.5
    z_value = ss.norm.interval(alpha)[1]
    
    minimum = (p1 - p2) - (z_value * sp_error)
    maximum = (p1 - p2) + (z_value * sp_error)
    
    return minimum, maximum
  

def pair_2p(n, cond1, cond2, alpha):
    """
    Function to get the confidence interval to estimate the difference between
    2 population averages in paired samples.
    
    Parameters:
    --------------------------
    n : int
        Sample sizes.
        
    cond1 : list
        Sample initial condition list.
    
    cond2 : list
        Sample final condition list.
        
    alpha : double, float
        Confidence level to estimate our unknown parameter (the difference between
        2 population averages).
        Each value should be in the range [0, 1].
        
    Returns:
    --------------------------
    interval : tuple
        The confidence interval, with the minimum value at [0] and the maximum value
        at [1] for the unknown parameter (the difference between
        2 population averages).
    
    """
    d1 = [cond1[x] - cond2[x] for x in range(n)]
    d21 = [x**2 for x in d1]
    sumd1 = sum(d1)
    sumd21 = sum(d21)
    d_ = sumd1 / n
    sd = ((sumd21 - (n * (d_**2))) / (n - 1))**0.5
    sd_error = sd / (n**0.5)
    
    if n <= 30:
        df = n - 1
        t_value = ss.t.interval(alpha, df)[1]
        
        minimum = d_ - (t_value * sd_error)
        maximum = d_ + (t_value * sd_error)
        
        return minimum, maximum
    
    else:
        z_value = ss.norm.interval(alpha)[1]
        
        minimum = d_ - (z_value * sd_error)
        maximum = d_ + (z_value * sd_error)
        
        return minimum, maximum
    
    

def n_avg(alpha, s, error):
    """
    Function to determine the appropiate sample size to estimate the average population.
    
    Parameters:
    --------------------------         
    alpha : float, double
        Confidence level to determine our sample size.
        Each value should be in the range [0, 1].
        
    s: float, double
        Pilot sample standard deviation, or population standar deviation.
        
    error: float, double
        Tolerable size error.
    
    Returns:
    --------------------------
    n : int
        The appropiate sample size to estimate the average population.
        This sample size is rounded.
    """
    z_value = ss.norm.interval(alpha)[1]
    
    n = ((z_value**2)*(s**2)) / (error**2)
    
    return round(n)

def n_avg_2p(alpha, s1, s2, error):
    """
    Function to determine the appropiate sample size to estimate the difference
    between 2 average populations.
    
    Parameters:
    --------------------------         
    alpha : float, double
        Confidence level to determine our sample size.
        Each value should be in the range [0, 1].
        
    s1 : float, double
        Pilot sample 1 standard deviation, or population standar deviation.
        
    s2 : float, double
        Pilot sample 2 standard deviation, or population standar deviation.
        
    error: float, double
        Tolerable size error.
    
    Returns:
    --------------------------
    n : int
        The appropiate sample size to estimate the he difference
        between 2 average populations.
        This sample size is rounded.
    """
    z_value = ss.norm.interval(alpha)[1]
    
    n = ((z_value**2)*(s1**2 + s2**2)) / (error**2)
    
    return round(n)



def n_prop(alpha, p, error):
    """
    Function to determine the appropiate sample size to estimate the population proportion.
    
    Parameters:
    --------------------------         
    alpha : float, double
        Confidence level to determine our sample size.
        Each value should be in the range [0, 1].
        
    p: float, double
        Sample proportion.
        Each value should be in the range [0, 1].
        
    error: float, double
        Tolerable size error.
    
    Returns:
    --------------------------
    n : int
        The appropiate sample size to estimate the population proportion.
        This sample size is rounded.
    """
    z_value = ss.norm.interval(alpha)[1]
    
    n = ((z_value**2) * (p * (1 - p))) / (error**2)
    
    return round(n)


def n_prop_2p(alpha, p1, p2, error):
    """
    Function to determine the appropiate sample size to estimate the 
    difference between 2 population proportions.
    
    Parameters:
    --------------------------         
    alpha : float, double
        Confidence level to determine our sample size.
        Each value should be in the range [0, 1].
        
    p1: float, double
        Sample proportion 1.
        Each value should be in the range [0, 1].
        
    p2: float, double
        Sample proportion 2.
        Each value should be in the range [0, 1].
        
    error: float, double
        Tolerable size error.
    
    Returns:
    --------------------------
    n : int
        The appropiate sample size to estimate the difference between 
        2 population proportions.
        This sample size is rounded.
    """
    z_value = ss.norm.interval(alpha)[1]
    
    n = ((z_value**2) * ((p1 * (1 - p1)) + ((p2 * (1 - p2))))) / (error**2)
    
    return round(n)
            
def f_2p(n1, n2, var1, var2, alpha):
    """
    """
    df1 = n1 - 1
    df2 = n2 - 1
    f_value = ss.f(df1, df2).interval(alpha)
    
    minimum = var1 / (f_value[1] * var2)
    maximum = var1 / (f_value[0] * var2)
    
    return minimum, maximum

def beta(x, y, alpha):
    """
    """
    df = len(x) - 1
    t_value = ss.t(df).interval(alpha)[1]
    
    minimum = linregr.slope(x, y) - (t_value * linregr.get_sb(x, y))
    maximum = linregr.slope(x, y) + (t_value * linregr.get_sb(x, y))
    
    return minimum, maximum
    

#------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------#