import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

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
    
#------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------#



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
    decision : boolean
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
    if tail == 'two':
        z_area = ss.norm.cdf(z)
        p = 2 * (1 - z_area)
        return p
    else :
        z_area = ss.norm.cdf(z)
        p = 1 - z_area
        return p
            
    
#------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------#
