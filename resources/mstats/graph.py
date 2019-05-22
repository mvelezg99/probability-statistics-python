# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 11:59:54 2019

@author: Miguel Ángel Vélez
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from mstats import linregr as lr

#------------------------------------------------------------------------------------------------#
#------------------------------------------ Graph -----------------------------------------------#
#------------------------------------------------------------------------------------------------#

def discrete(distr):
    """
    """
    x_values = np.arange(distr.ppf(0.01), distr.ppf(0.99))
    y_values = distr.pmf(x_values)
    
    plt.plot(x_values, y_values)
    plt.show()
    
def discrete_range(distr, a, b):
    """
    """
    x_values = np.arange(distr.ppf(0.01), distr.ppf(0.99))
    x_values2 = np.arange(a, (b + 1))
    y_values = distr.pmf(x_values)
    y_values2 = distr.pmf(x_values2)
    
    plt.plot(x_values, y_values)
    plt.vlines(x_values2, 0, y_values2, colors='b', lw=5, alpha=0.5)
    plt.show()
    
def continuous(distr):
    """
    """
    x_values = np.linspace(distr.ppf(0.01), distr.ppf(0.99), 100)
    y_values = distr.pdf(x_values)
    
    plt.plot(x_values, y_values)
    plt.show()
    
def continuous_range(distr, a, b):
    """
    """
    x_values = np.linspace(distr.ppf(0.01), distr.ppf(0.99), 100)
    x_values2 = np.linspace(a, b)
    y_values = distr.pdf(x_values)
    y_values2 = distr.pdf(x_values2)
    
    plt.plot(x_values, y_values)
    plt.vlines(x_values2, 0, y_values2, colors='b', lw=5, alpha=0.5)
    plt.show()
    
def hypothesis(distr, test, sign, tail):
    """
    
    Parameters:
    --------------------------
    
    Returns:
    --------------------------
    """
    alpha = 1 - sign
    x_values = np.linspace(distr.ppf(0.001), distr.ppf(0.999), 100)
    y_values = distr.pdf(x_values)    
    plt.plot(x_values, y_values, 'black')
    
    if (tail == 'two'):
        crit = distr.interval(alpha)
        x_values2 = np.append(np.linspace(distr.ppf(0.001), crit[0]), 
                              np.linspace(crit[1], distr.ppf(0.999)))
        y_values2 = distr.pdf(x_values2)
        
        x_values3 = np.linspace(crit[0] + 0.1, crit[1] - 0.1)
        y_values3 = distr.pdf(x_values3)
        
        plt.vlines(crit[0], 0, distr.pdf(crit[0]), colors='black', lw=2, alpha=1)
        plt.vlines(crit[1], 0, distr.pdf(crit[1]), colors='black', lw=2, alpha=1)
        
        plt.vlines(x_values2, 0, y_values2, colors='gray', lw=5, alpha=0.5)
        plt.vlines(x_values3, 0, y_values3, colors='khaki', lw=5, alpha=0.5)
        
        plt.annotate(
                "Critical value: {}".format(round(crit[0], 2)),
                xy=(crit[0], distr.pdf(crit[0])),
                xytext=(50, 30),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='darkred', alpha=0.5),
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0')
                )
        plt.annotate(
                "Critical value: {}".format(round(crit[1], 2)),
                xy=(crit[1], distr.pdf(crit[1])),
                xytext=(50, 30),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='darkred', alpha=0.5),
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0')
                )
        
    if (tail == 'left'):
        crit = distr.ppf(1 - alpha)
        
        x_values2 = np.linspace(distr.ppf(0.001), crit)
        y_values2 = distr.pdf(x_values2)
        
        x_values3 = np.linspace(crit + 0.1, distr.ppf(0.999))
        y_values3 = distr.pdf(x_values3)
        
        plt.vlines(x_values2, 0, y_values2, colors='gray', lw=5, alpha=0.5)
        plt.vlines(x_values3, 0, y_values3, colors='khaki', lw=5, alpha=0.5)
        
        plt.annotate(
            "Critical value: {}".format(round(crit, 2)),
            xy=(crit, distr.pdf(crit)),
            xytext=(50, 30),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='darkred', alpha=0.5),
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0')
            )
        
    if (tail == 'right'):
        crit = distr.ppf(alpha)
        
        x_values2 = np.linspace(crit, distr.ppf(0.999))
        y_values2 = distr.pdf(x_values2)
        
        x_values3 = np.linspace(distr.ppf(0.001), crit - 0.1)
        y_values3 = distr.pdf(x_values3)
        
        plt.vlines(x_values2, 0, y_values2, colors='gray', lw=5, alpha=0.5)
        plt.vlines(x_values3, 0, y_values3, colors='khaki', lw=5, alpha=0.5)
        
        plt.annotate(
            "Critical value: {}".format(round(crit, 2)),
            xy=(crit, distr.pdf(crit)),
            xytext=(50, 30),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='darkred', alpha=0.5),
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0')
            )
            
            
            
        
    plt.vlines(test, 0, distr.pdf(test), colors='darkred', lw=2, alpha=1)
    plt.annotate(
            "Statistical test: {}".format(round(test, 2)),
            xy=(test, distr.pdf(test)),
            xytext=(60, 80),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0')
            )
    
    plt.subplots_adjust(left=0.125*2, bottom=None, right=0.9*2, top=None,
            wspace=None, hspace=None)
    
    gray_patch = mpatches.Patch(color='gray', label='Reject zone')
    khaki_patch = mpatches.Patch(color='#F7F2C5', label='No reject zone')
  
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.,
               handles=[gray_patch, khaki_patch])
               
    
    plt.show()
    
    
def linregr(x, y):
    """
    """
    plt.scatter(x, y, color='seagreen')
    fx = [(lr.regression(x, y))(i) for i in x]
    plt.plot(x, fx, color='dimgray')
    plt.title(r'$\hat Y = {} + {}x$'.format(round(lr.intercept(x, y), 2), round(lr.slope(x, y), 2)),
              fontsize=17)

    
    plt.show()
    
def trendline(y, t):
    """
    """
    plt.scatter(t, y, color='lightblue')
    ft = [(lr.regression(t, y))(i) for i in t]
    plt.plot(t, ft, color='dimgray')
    
    plt.show()
    

def timeserie(y, t):
    """
    """
    plt.scatter(t, y)
    plt.plot(t, y)
    
    plt.ylabel("Y values")
    plt.xlabel("Time values")
    
    plt.title("Time Serie Graph")
    
    plt.show()

#------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------#