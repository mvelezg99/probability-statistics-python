# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:33:01 2019

@author: Miguel Angel Velez
"""

import numpy as np
import scipy.stats as ss

#------------------------------------------------------------------------------------------------#
#---------------------------------- Probability Distributions -----------------------------------#
#------------------------------------------------------------------------------------------------#

def binom_pmf(n, pi, x):
    """
    """
    px = ss.binom(n, pi).pmf(x)
    return px

def binom_cdf(n, pi, x):
    """
    """
    px = ss.binom(n, pi).cdf(x)
    return px

def hypergeom_pmf(N, r, n, x):
    """
    """
    px = ss.hypergeom(N, r, n).pmf(x)
    return px

def hypergeom_cdf(N, r, n, x):
    """
    """
    px = ss.hypergeom(N, r, n).cdf(x)
    return px

def poisson_pmf(mu, x):
    """
    """
    px = ss.poisson(mu).pmf(x)
    return px

def poisson_cdf(mu, x):
    """
    """
    px = ss.poisson(mu).cdf(x)
    return px

def expon_cdf(mu, x): #mu = mu
    """
    Función para calcular la función de probabilidad acumulada de la distribución exponencial continua
    """
    px = 1 - np.exp(-mu * x)
    return px

def uniform_cdf(a, b, x):
    """
    """
    width = b - a
    px = ss.uniform(a, width).cdf(x)
    return px

def norm_cdf(mu, sigma, x):
    """
    """
    px = ss.norm(mu, sigma).cdf(x)
    return px

def sample_norm_cdf(mu, sigma, n, x):
    """
    """
    error = sigma / (n**0.5)
    px = ss.norm(mu, error).cdf(x)
    return px

def sample_prop_cdf(n, pi, p):
    """
    """
    error = ((pi * (1 - pi)) / n)**0.5
    px = ss.norm(pi, error).cdf(p)
    return px

#------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------#