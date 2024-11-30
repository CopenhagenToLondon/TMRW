import yfinance as yf
import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt

from datetime import datetime, date, timedelta, timezone
import sqlalchemy as sa

from scipy.stats import norm
from scipy.stats import linregress
from scipy import integrate
from statsmodels.tsa.stattools import adfuller


def mk_test(ts, mode = 'simple', window = 50, alpha = 0.00001):
    """
    Mann-Kendall test for trend
    Detects linear trends (in pair with cox_stuart)
    Optimal window = 50

    Input:
        x:   a vector of data
        alpha: significance level (0.001 default)
        window: last n values, for finding trend
        mode: 
            - 'full' - if you need to know trend direction; 
            - 'simple' - if just trend existance.

    Output:
        trend: tells the trend (increasing, decreasing or no trend)
        h: True (if trend is present) or False (if trend is absence)
        p: p value of the significance test
        z: normalized test statistics 

    """
    x = ts[-window:]
    n = len(x)

    # calculate S 
    s = 0
    for k in range(n-1):
        for j in range(k+1,n):
            s += np.sign(x[j] - x[k])
    #s = [-1 if x[j] < x[k] else 1 for j in xrange(k+1,n) for k in xrange(n-1)]

    # calculate the unique data
    unique_x = np.unique(x)
    g = len(unique_x)
    # calculate the var(s)
    n = float(n)
    if n == g: # there is no tie
        var_s = (n*(n-1)*(2*n+5))/18
        #print (var_s)
    else: # there are some ties in data
        tp = np.zeros(unique_x.shape)
        for i in range(len(unique_x)):
            tp[i] = sum(unique_x[i] == x)
        var_s = (n*(n-1)*(2*n+5) + np.sum(tp*(tp-1)*(2*tp+5)))/18

    if s>0:
        z = (s - 1)/np.sqrt(var_s)
    elif s == 0:
        z = 0
    elif s<0:
        z = (s + 1)/np.sqrt(var_s)

    # calculate the p_value
    p = 2*(1-norm.cdf(abs(z))) # two tail test

    h = abs(z) > norm.ppf(1-alpha/2) 

    if (z<0) and h:
        trend = 'decreasing'
    elif (z>0) and h:
        trend = 'increasing'
    else:
        trend = 'no trend'
    if mode == 'full':
        return trend, h, p, z
    else:
        return h, abs(z)


def cox_stuart(timeseries, window = 50, alpha = 0.0001, debug = False):
    """
    Cox-Stuart criterion
    H0: trend exists
    H1: otherwise

    Detects linear trends
    Optimal window = 50
    """
    n = window
    idx = np.arange(1,n+1)
    X = pd.Series(timeseries[-n:], index=idx)
    
    S1 = [(n-2*i) if X[i] <= X[n-i+1] else 0 for i in range(1,n//2)]
    n = float(n)
    S1_ = (sum(S1) - n**2 / 8) / math.sqrt(n*(n**2-1)/24)
    u = norm.ppf(1-alpha/2)
    if debug:
        print ('|S1*|:', abs(S1_))
        print ("u:",u)
    
    return abs(S1_) > u, abs(S1_) #H0 accept

def abbe_criterion(timeseries, window = 40, alpha = .00001, debug = False):

    """
    Abbe-Linnik criterion
    Detects exponential trend (in pair with autocorrelation)
    If window >50 => too much True
    """
    if len(timeseries) < window:
        return False
    series = pd.Series(timeseries)
    a = series[-window:]
    mean = a.mean()
    

    X = zip(a[:-1],a[1:])
    s1 = sum([math.pow((x[0]-x[1]),2) for x in X])
    s2 = sum([pow((x-mean),2) for x in a])
    q = 0.5 * s1 / s2
    q_a = 0.6814
    n = len(a)
    Q = -(1-q)*math.sqrt((2*n+1)/(2-pow((1-q),2)))
    u = norm.ppf(alpha) #kobzar page 26
    if debug:
        print ('mean:', mean)
        print ('q:', q)
        print ('Q*:', Q)
        print ('U_alpha-1:', u)
    return Q < u, abs(Q)


def autocorrelation(timeseries, window = 50, alpha = .00001, debug = False):
    """
    Detects exponential trend

    """
    n = window
    idx = np.arange(1,n+1)
    series = pd.Series(timeseries[-n:], index=idx)

    X = zip(series[:-1],series[1:])
    sqr_sum = pow(sum(series),2)
    n_f = float(n)
    r = (n_f * sum([x[0]*x[1] for x in X]) - sqr_sum + n_f*series[1]*series[n]) / \
        (n_f * sum([x**2 for x in series]) - sqr_sum)
    
    r_ = abs(r + 1./(n_f-1.)) / math.sqrt(n_f*(n_f-3)/(n_f+1)/pow((n_f-1),2))
    u = norm.ppf(1-alpha/2)
    
    return abs(r_) > u, abs(r_)







class mathematics:
    
    def __init__(self):
        self.solutions = []
    
    def solver(self):
        for i in range(-100,100):
            i = i/10
            if self.f(i) == 0:
                self.solutions.append(i)
        return self.solutions
    
    def integral_1(self,a,b):
        if round(integrate.quad(self.f, a, b)[0],2) == 1.0:
            integ = True
        else:
            integ = False
        return integ
    
    def positivity(self,a,b):
        a = round(a)
        b = round(b)
        pos = True
        for i in range(a,b):
            if self.f(i) < 0:
                pos = False       
        return pos
    
    def is_prime(n: int):
        if n <= 3:
            return n > 1
        if n % 2 == 0 or n % 3 == 0:
            return False
        limit = int(math.sqrt(n))
        for i in range(5, limit+1, 6):
            if n % i == 0 or n % (i+2) == 0:
                return False
        return True
    
    def modularity(n: int):
        modul = []
        for i in range(1,n):
            if n%i == 0:
                modul.append(i)
            else:
                next
        return modul
    
    def func_cont(f,a,b):
        factor = 1000
        interval = []
        for i in range(factor*100):
            interval.append(random.randint(a*factor,b*factor)/factor)
        interval.sort
        
        cont = True
        for i in range(len(interval)):
            if f(interval[i])-f(interval[i]-0.00001) > 0.0005:
                cont = False
            if f(interval[i]+0.00001)-f(interval[i]) > 0.0005:
                cont = False
        return cont
    
    def dimensionality(x):
        k = 0
        if type(x) == list:
            if type(x[1]) == list:
                for i in range(len(x)):
                    k = k + len(x[i])
                
            elif type(x[1]) != list:
                k = len(x)
                
        if type(x) == pd.Series:
            k = len(x)
        if type(x) == pd.DataFrame:
            k = len(x) * len(x.columns)       
                
        return k
    
    def cardinality(x):
        return len(x)
        
    def f(self,x):
        if x <= 1.3:
            y = 3*x**2+5*x-2
        elif x > 1.3 and x <= 1.5:
            y = 2*x -10
        elif x > 1.5:
            y = 5*x -2       
        return y
    
    def f2(self,x):
        y = 3*x**2+5*x-2
        return y
    
    def sigma_algebra(A):
        S = []
        colec = []
        
        if str([]) not in S:
            S.append([])
    
        for i in A: 
            
            if type(i) == int or type(i) == float or type(i) == str:
                colec.append(i)
            elif type(i) != int and type(i) != float and type(i) != str:
                if type(i) == list:
                    S.append([i])
    
                if len(colec) > 0:
                    S.append(colec)
                colec = []
            if i == A[-1] and len(colec) > 0 and type(A[-1]) != list and len(colec) < len(A):
                S.append(colec)
            
    
        for a in A:
            l = [e for e in A if e != a]
            S.append(l)
    
        if str(A) not in S:
            S.append(A)
    
    
        return pd.DataFrame(S)
          