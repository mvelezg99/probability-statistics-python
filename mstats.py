import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt


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

def convert_z_norm(z, mu, rho):
    """
    Función para convertir una Z en una distribución normal cualquiera al valor correspondiente en la
    distribución normal estándar
    """
    z_converted = ((z - mu) / rho )
    return z_converted

def convert_z_muestral(z, mu, rho, n):
    """
    Función para convertir una Z en una distribución normal cualquiera al valor correspondiente en la
    distribución normal estándar, para distribuciones muestrales
    """
    z_converted = ((z - mu) / (rho / n**0.5))
    return z_converted


