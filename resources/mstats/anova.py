# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:04:29 2019

@author: Miguel Angel Velez
"""

import numpy as np
import scipy.stats as ss

#------------------------------------------------------------------------------------------------#
#------------------------------------------- ANOVA ----------------------------------------------#
#------------------------------------------------------------------------------------------------#

def get_sct(*args):
    """
    """
    arr_tot = np.concatenate(args)
    big_mean = np.mean(arr_tot)
    sct = sum([(x - big_mean)**2 for x in arr_tot])
    return sct
    
def get_sctr(*args):
    """
    """
    arr_tot = np.concatenate(args)
    big_mean = np.mean(arr_tot)
    sctr = sum([(len(x) * ((np.mean(x) - big_mean)**2)) for x in args])
    return sctr
   
    
def get_sce(*args):
    """
    """        
    sce = 0
    
    for x in args:
        for xj in x:
            sce += (xj - np.mean(x))**2
    return sce

def get_sce2(*args):
    """
    """
    sct = get_sct(*args)
    sctr = get_sctr(*args)
    scbl = get_scbl(*args)
    
    sce = sct - sctr - scbl
    return sce

def get_scbl(*args):
    """
    """
    lists = args
    blocks = list(map(list, zip(*lists)))
    arr_tot = np.concatenate(blocks)
    big_mean = np.mean(arr_tot)
    scbl = sum([(len(x) * ((np.mean(x) - big_mean)**2)) for x in blocks])
    
    return scbl
    

def get_cmt(*args):
    """
    """
    arr_tot = np.concatenate(args)
    df = len(arr_tot) - 1
    sct = get_sct(*args)
    cmt = sct / df
    return cmt

def get_cmtr(*args):
    """
    """
    sctr = get_sctr(*args)
    df = len(args) - 1
    cmtr = sctr / df
    return cmtr

def get_cme(*args):
    """
    """
    arr_tot = np.concatenate(args)
    sce = get_sce(*args)
    df = len(arr_tot) - len(args)
    cme = sce / df
    return cme

def get_cme2(*args):
    """
    """
    sce = get_sce2(*args)
    df = (len(args) - 1) * (len(args[0]) - 1)
    cme = sce / df
    return cme

def get_cmbl(*args):
    """
    """
    scbl = get_scbl(*args)
    df = len(args[0]) - 1
    cmbl = scbl / df
    return cmbl
    
def get_fratio(*args):
    """
    """
    cmtr = get_cmtr(*args)
    cme = get_cme(*args)
    fratio = cmtr / cme
    return fratio

def get_fratio2(*args):
    """
    """
    cmtr = get_cmtr(*args)
    cme = get_cme2(*args)
    fratio = cmtr / cme
    return fratio

def get_fratio2_cmbl(*args):
    """
    """
    cmbl = get_cmbl(*args)
    cme = get_cme2(*args)
    fratio = cmbl / cme
    return fratio

def get_dms(*args, sign):
    """
    """
    cme = get_cme(*args)
    
    
    
    for x in range(len(args)):
        for xj in range(x + 1, len(args)):
            if(len(args[x]) == len(args[xj])):
                fratio = ss.f.ppf((1 - sign), 1, (len(np.concatenate(args)) - len(args)))
                dms = (((1 / len(args[x])) + (1 / len(args[xj]))) * cme * fratio)**0.5
                x_mean = np.mean(args[x])
                xj_mean = np.mean(args[xj])
                x_diff = abs(x_mean - xj_mean)
                if (x_diff > dms):
                    print("|x_{} - x_{}| = |{} - {}| = {} > {}".format(x+1, xj+1,
                          x_mean, xj_mean, x_diff, dms))
                else:
                    print("|x_{} - x_{}| = |{} - {}| = {} <= {}".format(x+1, xj+1,
                          x_mean, xj_mean, x_diff, dms))
            else:
                fratio = ss.f.ppf((1 - sign), (len(args) - 1), (len(np.concatenate(args)) - len(args)))
                dms = (((1 / len(args[x])) + (1 / len(args[xj]))) * cme * fratio)**0.5
                x_mean = np.mean(args[x])
                xj_mean = np.mean(args[xj])
                x_diff = abs(x_mean - xj_mean)
                if (x_diff > dms):
                    print("|x_{} - x_{}| = |{} - {}| = {} > {}".format(x+1, xj+1,
                          x_mean, xj_mean, x_diff, dms))
                else:
                    print("|x_{} - x_{}| = |{} - {}| = {} <= {}".format(x+1, xj+1,
                          x_mean, xj_mean, x_diff, dms))
                        
#------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------#

