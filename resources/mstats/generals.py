# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:36:13 2019

@author: Miguel Angel Velez
"""

#------------------------------------------------------------------------------------------------#
#------------------------------------------ Generals --------------------------------------------#
#------------------------------------------------------------------------------------------------#


def get_z(x, mu, sd, **kwargs):
    """
    Function to get the Z value (standard deviations number under or over the average)
    correspondent of the x value (parameter) in the standard normal distribution.
    
    Parameters:
    --------------------------
    x : double, float
        Specific value from the random variable.
    
    mu : double, float
        Population average given.
        
    sd : double, float
        Population or sample standard deviation given.
        
    n : int, optional
        Sample size, if you are working with sample distributions. (default=None)
        
    Returns:
    --------------------------
    z : float, double
        The Z value (standard deviations number under or over the average)
        correspondent of the x value (parameter) in the standard normal distribution.
    """
    if not kwargs:
        z = ((x - mu) / sd)
        return z
    else:
        n = kwargs.get('n', None)
        if (n <= 30):
            print("The sample size must be greater than 30.")
        else:
            z = ((x - mu) / (sd / n**0.5))
            return z
        

def get_z_2p(x_1, x_2, s1, s2, n1, n2):
    """
    
    """
    if (n1 <= 30 or n2 <= 30):
        print("The sample sizes must be greater than 30.")
    else:
        s_error = ((s1**2 / n1) + (s2**2 / n2))**0.5
        
        z = (x_1 - x_2) / s_error
        return z
    

def get_t(x, mu, s, n):
    """
    Function to get the T value correspondent of the X value in the t-student
    distribution.
    
    Parameters:
    --------------------------
    x : double, float
        Specific value from the random variable.
    
    mu : double, float
        Population average given.
        
    sd : double, float
        Sample standard deviation given.
        
    n : int, optional
        Sample size, less than 30.
        
    Returns:
    --------------------------
    t : float, double
        The T value correspondent of the X value in the t-student.
    """
    if n > 30:
        print("The sample size must be less than 30.")
    else:
        t = (x - mu) / (s / (n**0.5))
        return t
    
    
def get_t_2p(x_1, x_2, s1, s2, n1, n2, var):
    """
    """
    if (n1 > 30 or n2 > 30):
        print("The sample sizes must be less than 30.")
    else:
        if (var is True):
            df = n1 + n2 - 2
            s2p = ((s1**2 * (n1 - 1)) + ((s2**2) * (n2 - 1))) / df
            s2p_error = ((s2p/n1) + (s2p/n2))**0.5
            
            t = (x_1 - x_2) / s2p_error
            return t
        else:
            s2p_error = ((s1**2 / n1) + (s2**2 / n2))**0.5
            t = (x_1 - x_2) / s2p_error
            return t
            
            

    
def get_z_prop(p, pi, n):
    """
    Function to get the Z value (standard deviations number under or over the average)
    correspondent of the x value (parameter) in the standard normal distribution.
    Applied in proportions.
    
    Parameters:
    --------------------------
    p : double, float
        "Succeses" observations sample proportion.
        Each value should be in the range [0, 1].
    
    pi : double, float
        Population proportion given.
        Each value should be in the range [0, 1].
        
    n : int
        Sample size.
        
    Returns:
    --------------------------
    z : float, double
        The Z value (standard deviations number under or over the average)
        correspondent of the x value (parameter) in the standard normal distribution.
    """
    error = ((pi * (1 - pi)) / n)**0.5
    z = (p - pi) / error
    return z

def get_z_2prop(p1, p2, n1, n2):
    """
    """
    sp_error = (((p1 * (1 - p1)) / n1) + ((p2 * (1 - p2)) / n2))**0.5
    
    z = (p1 - p2) / sp_error
    return z


def get_df_var(n1, n2, s1, s2):
    """
    """
    df = ((((s1**2) / n1) + ((s2**2) / n2))**2) / (((((s1**2) / n1)**2) / 
            (n1 - 1)) + ((((s2**2) / n2)**2) / (n2 - 1)))
    return df


def get_d_pair(n, cond1, cond2):
    """
    """
    d1 = [cond1[x] - cond2[x] for x in range(n)]
    d21 = [x**2 for x in d1]
    sumd1 = sum(d1)
    sumd21 = sum(d21)
    d_ = sumd1 / n
    sd = ((sumd21 - (n * (d_**2))) / (n - 1))**0.5
    sd_error = sd / (n**0.5)
    
    d = d_ / sd_error
    return d

def get_f_2p(var1, var2):
    """
    """
    f = var1 / var2
    return f

#------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------#