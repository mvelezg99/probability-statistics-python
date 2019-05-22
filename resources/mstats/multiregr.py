# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 15:18:27 2019

@author: Miguel Ángel Vélez

"""

import numpy as np

#------------------------------------------------------------------------------------------------#
#------------------------------------- Multiple Regression --------------------------------------#
#------------------------------------------------------------------------------------------------#

#y1 = [15, 17, 13, 23, 16, 21, 14, 20, 24, 17, 16, 18, 23, 15, 16]
##
#x1 = [10, 12, 8, 17, 10, 15, 10, 14, 19, 10, 11, 13, 16, 10, 12]
#x2 = [2.4, 2.72, 2.08, 3.68, 2.56, 3.36, 2.24, 3.2, 3.84, 2.72, 2.07, 2.33, 2.98, 1.94, 2.17]

def beta_hat(y, *args):
    """
    """
    x = list(args)
    
    ones = [1] * (len(y))
    X = np.matrix([ones] + x).transpose()
    
    #print(np.dot(transpose_X, X))
    # print(xs * transpose_x)
    
    #print(
            #np.dot(np.linalg.inv(np.dot(X.transpose(), X)), np.dot(y, X.transpose()))
            #np.dot(y, np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose()))
            #np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose()), y).tolist()[0]
            #)
            
    beta_hat = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose()), y).tolist()[0]
    
    return beta_hat

def regression(y, *args):
    """
    """
    betas = beta_hat(y, *args)
    model = lambda *xs : betas[0] + sum([(betas[i] * xs[i - 1]) for i in range(1, len(betas))])
    model.shape = "ÿ = {}".format(round(betas[0], 2))
    for i in range(1, len(betas)):
        model.shape += (" + {}X{}".format(round(betas[i], 2), i))
    return model

def y_hat(y, *args):
    """
    """
    model = regression(y, *args)
    
    length_args = len(args[0])
    length_parent = len(args)
    
    y_hat = []
    
    for i in range(0, length_args):
        aux = []
        #print("aux: ", aux)
        for j in range(0, length_parent):
            #print("args: ", args[j][i])
            aux.append(args[j][i])
            
        #print("*aux: ", *aux)
        y_hat.append(model(*aux))
        #print("yhat: ", y_hat)
        
    return y_hat
    

#print(y_hat(y1, x1, x2))

def get_se(y, *args):
    """
    """
    y_hats = y_hat(y, *args)
    
    n = len(y)
    k = len(args)
    
    se = (sum([(y[i] - y_hats[i])**2 for i in range(n)]) / (n - k - 1))**0.5
    return se

#print(get_se(y1, x1, x2))

def get_scr(y, *args):
    """
    """
    y_hats = y_hat(y, *args)
    n = len(y)
    mean_y = np.mean(y)
    
    scr = sum([(y_hats[i] - mean_y)**2 for i in range(n)])
    return scr

def get_sct(y, *args):
    """
    """
    mean_y = np.mean(y)
    n = len(y)
    sct = sum([(y[i] - mean_y)**2 for i in range(n)])
    return sct

def get_sce(y, *args):
    """
    """
    y_hats = y_hat(y, *args)
    n = len(y)
    
    sce = sum([(y[i] - y_hats[i])**2 for i in range(n)])
    return sce

def get_r2(y, *args):
    """
    """
    r2 = get_scr(y, *args) / get_sct(y, *args)
    return r2

def get_r2_(y, *args):
    """
    """
    n = len(y)
    k = len(args)
    
    r2_ = 1 - (1 - get_r2(y, *args)) * ((n - 1) / (n - k -1))
    return r2_

def get_cmr(y, *args):
    """
    """
    k = len(args)
    cmr = get_scr(y, *args) / k
    return cmr

def get_cme(y, *args):
    """
    """
    n = len(y)
    k = len(args)
    cme = get_sce(y, *args) / (n - k - 1)
    return cme

def get_fratio(y, *args):
    """
    """
    fratio = get_cmr(y, *args) / get_cme(y, *args)
    return fratio

def matrix_corr(y, *args):
    """
    """
    matrix = np.corrcoef(y, args)
    return matrix

def get_sr(rij, n):
    """
    """
    sr = ((1 - (rij)**2) / (n - 2))**0.5
    return sr

def t_multicollinearity(rij, sr):
    """
    """
    t = rij / sr
    return t

def t_regression(b, beta, sb):
    """
    """
    t = (b - beta) / sb
    return t

def get_sbs(y, *x):
    """
    """
    xs = list(x)
    ones = [1] * len(y)
    X = np.matrix([ones] + xs).transpose()
    
    X_matrix = np.linalg.inv(np.dot(X.transpose(), X))
    
    cme = get_cme(y, *x)
    
    C = cme * X_matrix
    
    sbs = [(C[i, i]**0.5) for i in range(len(x) + 1)]
    
    return sbs
    
#print(get_sbs(y1, x1, x2))

#print(matrix_corr(y1, x1, x2))
#print(get_sr(matrix_corr(y1, x1, x2)[1, 2], len(y1)))

#------------------------------------------------------------------------------------------------#
