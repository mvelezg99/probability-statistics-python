import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------------------------#
#------------------------------------------ Generals --------------------------------------------#
#------------------------------------------------------------------------------------------------#
class generals:
    """
    generals Class contain the general statistics methods to work with probabilities,
    distributions, confidence intervals and hypothesis testing.
    
    Methods:
    --------------------------
    get_z(x, mu ,sd, n=None)
        Function to get the Z value.
    """

            
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
        

      
    
    
#------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------#



#------------------------------------------------------------------------------------------------#
#---------------------------------- Probability Distributions -----------------------------------#
#------------------------------------------------------------------------------------------------#

class distributions:

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
class intervals:
    """
    interval Class contain the methods to get the confidence intervals to
    estimate some unknown parameter.
    
    Methods:
    --------------------------
    norm(n, x_ ,sd, alpha)
        Function to get the confidence interval of average population in large
        samples.
    """
    
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
            
            
        
            



#------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------#


#------------------------------------------------------------------------------------------------#
#------------------------------------- Hypothesis Testing ---------------------------------------#
#------------------------------------------------------------------------------------------------#
class hypothesis:
    """
    hypothesis Class contain the methods to work on the hypothesis testing.
    
    Methods:
    --------------------------
    crit_val_norm(sign, tail)
        Function to calculate the critical values in the standard normal distribution.
    """
    
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
        
    
    def crit_val_f(df1, df2, sign):
        """
        """
        alpha = 1 - sign
        crit_val = ss.f.ppf(alpha, df1, df2)
        return crit_val
                
        
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
        decision : bool
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
        z_area = ss.norm.cdf(abs(z))
        
        if tail == 'two':
            p = 2 * (1 - z_area)
            return p
        
        else:
            p = 1 - z_area
            return p

            
    
#------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------#
            
#------------------------------------------------------------------------------------------------#
#------------------------------------------- ANOVA ----------------------------------------------#
#------------------------------------------------------------------------------------------------#        
class anova:
    
    
    def get_sct(*args):
        """
        """
        arr_tot = np.concatenate(args)
        big_mean = np.mean(arr_tot)
        sct = sum([(x - big_mean)**2 for x in arr_tot])
        return sct
        
    
        
        #d1 = [cond1[x] - cond2[x] for x in range(n)]
        #d21 = [x**2 for x in d1]
        
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
        sct = anova.get_sct(*args)
        sctr = anova.get_sctr(*args)
        scbl = anova.get_scbl(*args)
        
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
        sct = anova.get_sct(*args)
        cmt = sct / df
        return cmt
    
    def get_cmtr(*args):
        """
        """
        sctr = anova.get_sctr(*args)
        df = len(args) - 1
        cmtr = sctr / df
        return cmtr
    
    def get_cme(*args):
        """
        """
        arr_tot = np.concatenate(args)
        sce = anova.get_sce(*args)
        df = len(arr_tot) - len(args)
        cme = sce / df
        return cme
    
    def get_cme2(*args):
        """
        """
        sce = anova.get_sce2(*args)
        df = (len(args) - 1) * (len(args[0]) - 1)
        cme = sce / df
        return cme
    
    def get_cmbl(*args):
        """
        """
        scbl = anova.get_scbl(*args)
        df = len(args[0]) - 1
        cmbl = scbl / df
        return cmbl
        
    def get_fratio(*args):
        """
        """
        cmtr = anova.get_cmtr(*args)
        cme = anova.get_cme(*args)
        fratio = cmtr / cme
        return fratio
    
    def get_fratio2(*args):
        """
        """
        cmtr = anova.get_cmtr(*args)
        cme = anova.get_cme2(*args)
        fratio = cmtr / cme
        return fratio
    
    def get_fratio2_cmbl(*args):
        """
        """
        cmbl = anova.get_cmbl(*args)
        cme = anova.get_cme2(*args)
        fratio = cmbl / cme
        return fratio
    
    def get_dms(*args, sign):
        """
        """
        cme = anova.get_cme(*args)
        
        
        for x in range(len(args)):
            for xj in range(x+1, len(args)):
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