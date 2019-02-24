import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------------------------#
#---------------------------------- Probability Distributions -----------------------------------#
#------------------------------------------------------------------------------------------------#

def graficar_discreta(distribucion, title):
    """
    Función para graficar la distribución discreta recibida como parámetro
    """
    valores_x = np.arange(distribucion.ppf(0.01), distribucion.ppf(0.99))
    fmp = distribucion.pmf(valores_x)
    plt.plot(valores_x, fmp, '-')
    plt.title(title)
    plt.ylabel("Probabilidad")
    plt.xlabel("Valores")
    plt.show()


    
def graficar_discreta_cdf(distribucion, a, b):
    """
    Función para graficar la sumatoria de la distribución discreta recibida como parámetro
    """
    valores_x = np.arange(distribucion.ppf(0.01), distribucion.ppf(0.99))
    x2 = np.arange(a,(b+1))
    fmp2 = distribucion.pmf(x2)
    fmp = distribucion.pmf(valores_x)
    plt.plot(valores_x, fmp)
    plt.vlines(x2, 0, fmp2, colors='b', lw=5, alpha=0.5)
    plt.title("Probabilidad encontrada")
    plt.ylabel("Probabilidad")
    plt.xlabel("Valores")
    plt.show()
    


def graficar_continua(distribucion, title):
    """
    Función para graficar la distribución continua recibida como parámetro
    """
    valores_x = np.linspace(distribucion.ppf(0.01), distribucion.ppf(0.99), 100)
    fp = distribucion.pdf(valores_x)
    plt.plot(valores_x, fp, '-')
    plt.title(title)
    plt.ylabel("Probabilidad")
    plt.xlabel("Valores")
    plt.show()
    


def graficar_continua_cdf(distribucion, a, b):
    """
    Función para graficar la integral de la distribución continua recibida como parámetro
    """
    valores_x = np.linspace(distribucion.ppf(0.01), distribucion.ppf(0.99), 100)
    x2 = np.linspace(a, b)
    fp2 = distribucion.pdf(x2)
    fp = distribucion.pdf(valores_x)
    plt.plot(valores_x, fp)
    plt.vlines(x2, 0, fp2, colors='b', lw=5, alpha=0.5)
    plt.title("Probabilidad encontrada")
    plt.ylabel("Probabilidad")
    plt.xlabel("Valores")
    plt.show()


    

def cdf_exponential(mu, x): #mu = mu
    """
    Función para calcular la función de probabilidad acumulada de la distribución exponencial continua
    """
    if x <= 0:
        cdf = 0
    else:
        cdf = 1 - np.exp(-mu * x)
    return cdf



def get_z_norm(z, mu, rho):
    """
    Función para obtener una Z en una distribución normal cualquiera al valor correspondiente en la
    distribución normal estándar
    """
    z_converted = ((z - mu) / rho )
    return z_converted



def get_z_muestral(z, mu, rho, n):
    """
    Función para convertir una Z en una distribución normal cualquiera al valor correspondiente en la
    distribución normal estándar, para distribuciones muestrales
    """
    z_converted = ((z - mu) / (rho / n**0.5))
    return z_converted

#------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------#



#------------------------------------------------------------------------------------------------#
#------------------------------------- Confidence Intervals -------------------------------------#
#------------------------------------------------------------------------------------------------#

def interval_norm(n, x_, sd, alpha):
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
    --------------------------
    
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
        
        minimum_value = x_ - (z_value * sd_error)
        maximum_value = x_ + (z_value * sd_error)
        
        return minimum_value, maximum_value
    

           
def interval_t(n, x_, s, alpha):
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
    --------------------------
    
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
        
        minimum_value = x_ - (t_value * s_error)
        maximum_value = x_ + (t_value * s_error)
        
        return minimum_value, maximum_value



def interval_prop(n, p, alpha):
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
    --------------------------
    
    Returns:
    --------------------------
    interval : tuple
        The confidence interval, with the minimum value at [0] and the maximum value
        at [1] for the unknown parameter (the population proportion).
    """   
    sp_error = ((p * (1 - p)) / n)**0.5
    z_value = ss.norm.interval(alpha)[1]
    
    minimum_value = p - (z_value * sp_error)
    maximum_value = p + (z_value * sp_error)
    
    return minimum_value, maximum_value



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
    --------------------------
    
    Returns:
    --------------------------
    n : int
        The appropiate sample size to estimate the average population.
        This sample size is rounded.
    """
    z_value = ss.norm.interval(alpha)[1]
    
    n = ((z_value**2)*(s**2)) / (error**2)
    
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
        Population proportion expected value.
        Each value should be in the range [0, 1].
        
    error: float, double
        Tolerable size error.
    --------------------------
    
    Returns:
    --------------------------
    n : int
        The appropiate sample size to estimate the population proortion.
        This sample size is rounded.
    """
    z_value = ss.norm.interval(alpha)[1]
    
    n = ((z_value**2) * (p * (1 - p))) / (error**2)
    
    return round(n)

#------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------#



