# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 09:14:28 2019

@author: Miguel Ángel Vélez
"""

import numpy as np
#------------------------------------------------------------------------------------------------#
#-------------------------------------- Linear Regression ---------------------------------------#
#------------------------------------------------------------------------------------------------#

def get_xy(x, y):
    """
    """
    n = len(x)
    xy = [x[i] * y[i] for i in range(n)]
    return xy

def get_x2(x):
    """
    """
    x2 = [x**2 for x in x]
    return x2

def get_sc(x):
    """
    """
    n = len(x)
    x2 = get_x2(x)
    sc = sum(x2) - (sum(x))**2 / n
    return sc

def get_spc(x, y):
    """
    """
    n = len(x)
    xy = get_xy(x,y)
    spc = sum(xy) - (sum(x) * sum(y)) / n
    return spc

def get_sce(x, y):
    """
    """
    sce = get_sc(y) - (get_spc(x,y)**2) / get_sc(x)
    return sce

def get_cme(x, y):
    """
    """
    cme = get_sce(x, y) / (len(x) - 2)
    return cme

def get_se(x, y):
    """
    """
    se = get_cme(x, y)**0.5
    return se

def get_sb(x, y):
    """
    """
    sb = get_se(x,y) / (get_sc(x)**0.5)
    return sb

def get_sr(x, y):
    """
    """
    sr = (1 - get_r2(x, y) / (len(x) - 2))**0.5
    return sr

def get_r(x, y):
    """
    """
    r = get_spc(x, y) / ((get_sc(x) * get_sc(y))**0.5)
    return r

def get_r2(x, y):
    """
    """
    r2 = get_r(x, y)**2
    return r2

def get_t_beta(x, y, beta=0):
    """
    """
    t = (slope(x, y) - beta) / get_sb(x, y)
    return t

def get_t_rho(x, y, rho=0):
    """
    """
    t = (get_r(x, y) - rho) / get_sr(x, y)
    return t

def slope(x, y):
    """
    """
    m = get_spc(x, y) / get_sc(x)
    return m

def intercept(x, y):
    """
    """
    b = np.mean(y) - slope(x, y) * np.mean(x)
    return b

def regression(x, y):
    """
    """
    model = lambda xi : intercept(x,y) + slope(x, y) * xi
    model.shape = "ÿ = {} + {}x".format(round(intercept(x, y), 2), round(slope(x, y), 2))
    return model

    

#------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------#
