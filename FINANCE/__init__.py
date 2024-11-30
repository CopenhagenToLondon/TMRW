"""
TMRW.FINANCE

Created on Mon Apr 15 22:44:02 2024

Version 1.0.0 bundled on Fri Aug 09 12:00:00 2024

@Author: Mark Daniel Balle Brezina
@Contributors: Magnus Faber, Jonathan Emil Balle Brezina
"""

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
from hmmlearn.hmm import GaussianHMM

import TMRW.DATA as td
import TMRW.MATH as tm

###############################################################################
#                        Slopes, returns and accelerations
#                        For calculating import 
#                        intra and inter day differences
###############################################################################

def slope(array):
    """
    """
    y = np.array(array)
    x = np.arange(len(y))
    slopes, intercept, r_value, p_value, std_err = linregress(x,y)
    return(slopes)

def returns(x, logs = False):
    """
    """
    if type(x) == list: # if you've entered a list
        _return = np.zeros(len(x))
        for i in range(len(x)):
            _return[i] = ((x[i]/x[i-1])-1)
    elif type(x) == pd.DataFrame: # if you've entered a dataframe
        _return = x.pct_change().dropna()   
    else: # if you've entered neither a list or a dataframe
        _return = x.pct_change().dropna()
    
    # if you want log returns
    if logs == True:
        _return = np.log(abs(_return))
    
    df = pd.DataFrame(data=_return, dtype=np.float64)
    df.columns = ["returns"]
    return(df)

def acceleration(x, logs = False):
    """
    """
    x = returns(x, logs)
    if type(x) == list:  # if you've entered a list
        acc = np.zeros(len(x))
        for i in range(len(x)):
            acc[i] = ((x[i]/x[i-1])-1)
    elif type(x) == pd.DataFrame: # if you've entered a dataframe
        acc = x.pct_change().dropna() 
    else: # if you've entered neither a list or a dataframe
        acc = x.pct_change().dropna()
        
     # if you want log acceleration  
    if logs == True:
        acc = np.log(abs(acc))
    
    df = pd.DataFrame(data=acc, dtype=np.float64)
    df.columns = ["acceleration"]
    return(df)

###############################################################################
#                        Moving averages and stds
#                        volume weighted averages and stds
#                        bollinger bands?
###############################################################################

# Time-weighted Moving Average
def twa(data, window = 20, typ = "sma", x = True):
    """
    time-weighted moving average
    """
    if typ == "sma": # simple moving average
        if len(data) < window+1:
            raise ValueError("not enough data")
            
        if x == True:
            
            df = []
            for i in range(len(data)):
                if i < window:
                    df.append(data[i])
                else:
                    df.append(np.mean(data[i-window:i]))

            df = pd.DataFrame(df, index = data.index)
            df = df.iloc[0:len(df), 0]
            
            return(df)
    
        elif x == False:

            df = []
            for i in range(len(data)):

                ma = np.zeros(window)

                for j in range(1,window):

                    if i < window:
                        ma[j] = data[j]


                    if i >= window:

                        ma[j] = np.mean(data[i-j:i])

                df.append(ma)

            df = pd.DataFrame(df, index = data.index)
            df = df.iloc[0:len(df), 1:len(df.columns)]
            
            return(df)
    
    elif typ == "ema": # exponential moving average

        df = pd.DataFrame(index=data.index, dtype=np.float64)
        df.loc[:,0] = data.ewm(span=window, min_periods=0, adjust=True, ignore_na=False).mean().values.flatten() # ????
    
        return(df)
      
#Time-Price Moving STD
def twstd(data, window = 20, x = True):
    """
    time-weighted moving standard deviation of prices
    also called volatility
    """
    if len(data) < window+1:
        raise ValueError("not enough data")

    if x == True:
        
        df = []
        for i in range(len(data)):
            if i < window:
                df.append(data[i])
            else:
                df.append(np.std(data[i-window:i]))

        df = pd.DataFrame(df, index = data.index)
        df = df.iloc[0:len(df), 0]
        
        return(df)
        
    elif x == False:  
        
        df = []
        for i in range(len(data)):

            mstd = np.zeros(window)

            for j in range(1,window):

                if i >= j:

                    mstd[j] = np.std(data[i-j:i])

            df.append(mstd)

        df = pd.DataFrame(df, index = data.index)
        df = df.iloc[0:len(df), 1:len(df.columns)]
    
        return(df)
    
# Volume-weighted Moving Average
def vwa(data, weights, window = 20):
    """
    volume-weighted moving average
    """
    
    vwa = np.zeros(window)
    vwa = list(vwa)
    for i in range(len(data)):
        
        if i < window:
            
            vwa[i] = data[i]
            
        else:
    
            v_sum = sum(weights[i-window:i])
            v_w_avg = 0
            for j in range(window):
                v_w_avg = v_w_avg + data[i-j] * (weights[i-j] / v_sum)

            vwa.append(v_w_avg)
    
    df = pd.DataFrame(vwa)
    return(df)

#Volume-weighted Moving STD
def vwstd(data, weights, window = 20):
    """
    volume-weighted moving standard deviations
    """
    
    v_w_avg = 0
    vws = np.zeros(window)
    vws = list(vws)
    for i in range(window, len(data)):
    
        v_sum = sum(weights[i-window:i])
        v_w_std = 0
        for j in range(window):
            v_w_std = v_w_std + ((data[i-j] - np.mean(data[i-window:i]))**2 )* (weights[i-j] / v_sum)

        vws.append(np.sqrt(v_w_std))
    
    df = pd.DataFrame(vws)
    return(df)

###############################################################################
#                        Technical indicators
#                        RSI, Stochastic Oscillator
#
###############################################################################

def RSI(data, window = 20):
    """
    
    """
    delta = data.diff().dropna() # Close_now - Close_yesterday
    delta = delta.reset_index(drop = True)

    u = pd.DataFrame(np.zeros(len(delta))) # make an array of 0s for the up returns
    u = u[0]
    d = u.copy() # make an array of 0s for the down returns   

    u[delta > 0] = delta[delta > 0] # for all the days where delta is up, transfer them to U
    d[delta < 0] = -delta[delta < 0] # for all the days where delta is down, transfer them to D

    u[u.index[window-1]] = np.mean( u[:window] ) #first value is sum of avg gains
    u = u.drop(u.index[:(window-1)]) #drop the days before the window opens

    d[d.index[window-1]] = np.mean( d[:window] ) #first value is sum of avg losses
    d = d.drop(d.index[:(window-1)]) #drop the days before the window opens

    RS = pd.DataFrame.ewm(u, com=window-1, adjust=False).mean() / pd.DataFrame.ewm(d, com=window-1, adjust=False).mean() # EMA(up) / EMA(down)

    RSI_ = 100 - (100 / (1 + RS))
    return(RSI_)

def FRSI(data, window = 20):
    """
    inverse fisher transform on RSI = 0.1*(rsi-50)
    fisher rsi = (np.exp(2*rsi)-1) / (np.exp(2*rsi)+1)
    """
    
    RSI_ = 0.1 * (RSI(data, window) - 50)
    F_RSI = (np.exp(2*RSI_)-1) / (np.exp(2*RSI_)+1)
    return(F_RSI)

def BB(data, window = 20, std = 2.5):
    """
    
    """
    mean_lst = np.zeros(len(data))
    std_lst = np.zeros(len(data))
    for i in range(0, window):
        mean_lst[i] = data[i]
        std_lst[i] = 0.05 * data[i]
    
    for i in range(window,len(data)):
        mean_lst[i] = np.mean(data[i-window:i])
        std_lst[i] = np.std(data[i-window:i])
        
    up = mean_lst + std * std_lst
    down = mean_lst - std * std_lst
    
    df = pd.DataFrame()
    df['Upper'] = up
    df['Lower'] = down
    return(df)

def STO(data, N=14, M=3):
    assert 'Low' in data.columns
    assert 'High' in data.columns
    assert 'Close' in data.columns
    
    data_ = pd.DataFrame()
    data_['low_N'] = data['Low'].rolling(N).min()
    data_['high_N'] = data['High'].rolling(N).max()
    data_['K'] = 100 * (data['Close'] - data_['low_N']) / \
        (data_['high_N'] - data_['low_N']) # The stochastic oscillator
    data_['D'] = data_['K'].rolling(M).mean() # the slow and smoothed K
    return data_

def ADX(high, low, close, lookback):
    
    df = pd.DataFrame()
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
    atr = tr.rolling(lookback).mean()
    
    plus_di = 100 * (plus_dm.ewm(alpha = 1/lookback).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha = 1/lookback).mean() / atr))
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = ((dx.shift(1) * (lookback - 1)) + dx) / lookback
    adx_smooth = adx.ewm(alpha = 1/lookback).mean()
    df['+DI'] = plus_di
    df['-DI'] = minus_di
    df['ADX'] = adx_smooth
    
    return df

def MACD(price, slow, fast, smooth):
    exp1 = price.ewm(span = fast, adjust = False).mean()
    exp2 = price.ewm(span = slow, adjust = False).mean()
    macd = pd.DataFrame(exp1 - exp2).rename(columns = {'Close':'macd'})
    signal = pd.DataFrame(macd.ewm(span = smooth, adjust = False).mean()).rename(columns = {'macd':'signal'})
    hist = pd.DataFrame(macd['macd'] - signal['signal']).rename(columns = {0:'hist'})
    frames =  [macd, signal, hist]
    df = pd.concat(frames, join = 'inner', axis = 1)
    return df

def SuperTrend(high, low, close, lookback, multiplier):
    # ATR
    df = pd.DataFrame()
    
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
    atr = tr.ewm(lookback).mean()
    
    # H/L AVG AND BASIC UPPER & LOWER BAND
    
    hl_avg = (high + low) / 2
    upper_band = (hl_avg + multiplier * atr).dropna()
    lower_band = (hl_avg - multiplier * atr).dropna()
    
    # FINAL UPPER BAND
    
    final_bands = pd.DataFrame(columns = ['upper', 'lower'])
    final_bands.iloc[:,0] = [x for x in upper_band - upper_band]
    final_bands.iloc[:,1] = final_bands.iloc[:,0]
    
    for i in range(len(final_bands)):
        if i == 0:
            final_bands.iloc[i,0] = 0
        else:
            if (upper_band[i] < final_bands.iloc[i-1,0]) | (close[i-1] > final_bands.iloc[i-1,0]):
                final_bands.iloc[i,0] = upper_band[i]
            else:
                final_bands.iloc[i,0] = final_bands.iloc[i-1,0]
    
    # FINAL LOWER BAND
    
    for i in range(len(final_bands)):
        if i == 0:
            final_bands.iloc[i, 1] = 0
        else:
            if (lower_band[i] > final_bands.iloc[i-1,1]) | (close[i-1] < final_bands.iloc[i-1,1]):
                final_bands.iloc[i,1] = lower_band[i]
            else:
                final_bands.iloc[i,1] = final_bands.iloc[i-1,1]
    
    # SUPERTREND
    
    supertrend = pd.DataFrame(columns = [f'supertrend_{lookback}'])
    supertrend.iloc[:,0] = [x for x in final_bands['upper'] - final_bands['upper']]
    
    for i in range(len(supertrend)):
        if i == 0:
            supertrend.iloc[i, 0] = 0
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 0] and close[i] < final_bands.iloc[i, 0]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 0]
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 0] and close[i] > final_bands.iloc[i, 0]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 1] and close[i] > final_bands.iloc[i, 1]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 1] and close[i] < final_bands.iloc[i, 1]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 0]
    
    supertrend = supertrend.set_index(upper_band.index)
    supertrend = supertrend.dropna()[1:]
    
    # ST UPTREND/DOWNTREND
    
    upt = []
    dt = []
    close = close.iloc[len(close) - len(supertrend):]

    for i in range(len(supertrend)):
        if close[i] > supertrend.iloc[i, 0]:
            upt.append(supertrend.iloc[i, 0])
            dt.append(np.nan)
        elif close[i] < supertrend.iloc[i, 0]:
            upt.append(np.nan)
            dt.append(supertrend.iloc[i, 0])
        else:
            upt.append(np.nan)
            dt.append(np.nan)
            
    st, upt, dt = pd.Series(supertrend.iloc[:, 0]), pd.Series(upt), pd.Series(dt)
    upt.index, dt.index = supertrend.index, supertrend.index
    
    df['ST'] = st
    df['UPT'] = upt
    df['DT'] = dt
    
    return df


###############################################################################
#                        Plots, graphs and the alike
#                        
#                        
###############################################################################

def PRICE_plot(data, MA = False):
    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 3))
    #ax = plt.gca()
    ax1.set_facecolor('dimgrey')
    ax1.plot(data.index, data.Close, color = "deepskyblue")
    #ax1.set_xticks([])
    plt.grid()
    plt.show()


def STO_plot(data):
    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 1))
    ax1.set_facecolor('dimgrey')
    
    ax1.plot(data.index, data.D, color = "deepskyblue")
    ax1.plot(data.index, np.repeat(80,len(data)), color = "white", linestyle = "dotted")
    ax1.plot(data.index, np.repeat(20,len(data)), color = "white", linestyle = "dotted")
    plt.show()

def ADX_plot(data, components = False):    
    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 1))
    ax1.set_facecolor('dimgrey')
    
    if components == False:    
        ax1.plot(data.index, data['ADX'], color = "deepskyblue")
    
    elif components == True:
        ax1.plot(data.index, data['-DI'], color = "r")
        ax1.plot(data.index, data['+DI'], color = "g")
    
    ax1.plot(data.index, np.repeat(40,len(data)), color = "white", linestyle = "dotted")
    ax1.plot(data.index, np.repeat(20,len(data)), color = "white", linestyle = "dotted")
    plt.show()

def MACD_plot(prices, macd, signal, hist):
    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 1))
    ax1.set_facecolor('dimgrey')

    for i in range(len(prices)):
        if str(hist[i])[0] == '-':
            ax1.bar(prices.index[i], hist[i], color = 'gold')
        else:
            ax1.bar(prices.index[i], hist[i], color = 'deepskyblue')
    
    plt.grid()
    plt.show()
    
def TRADE_plot(strategy):
    fig, (ax1) = plt.subplots(1, 1, figsize=(23, 12))
    ax1.set_facecolor('dimgrey')
    
    ax1.plot(strategy.index, strategy['Close'], linewidth = 2, color = "deepskyblue")
    
    buy_price = strategy_df[strategy_df['position'] == 1]
    sell_price = strategy_df[strategy_df['position'] == -1]
    short_price = strategy_df[strategy_df['position'] == -2]
    
    ax1.plot(buy_price.index, buy_price['Close'], marker = 'o', color = 'lime', markersize = 8, linewidth = 0)
    ax1.plot(sell_price.index, sell_price['Close'], marker = 'X', color = 'tomato', markersize = 8, linewidth = 0)
    ax1.plot(short_price.index, short_price['Close'], marker = 'X', color = 'purple', markersize = 8, linewidth = 0)
    plt.grid()
    plt.show()

# very useful   
def SUM_plot(data, strategy):
    fig, (ax1) = plt.subplots(1, 1, figsize=(23, 12))
    ax1.set_facecolor('dimgrey')
    
    rets = data.Close / data.Close[0]
    
    #ax1.plot(rets.index, rets, color = 'deepskyblue', linewidth = 2)
    #ax1.plot(data.index, rets, color = 'deepskyblue', linewidth = 2)
    ax1.plot(strategy.index, np.repeat(1, len(strategy)), color = 'r', linewidth = 2)
    ax1.plot(strategy.index, strategy.acc, color = 'w', linewidth = 2)
    plt.grid()
    plt.show()     

###############################################################################
#                        Portfolio level important functions
#                        VaR, CVaR, Drawdown, Annualized returns, etc.
#                        Not used for trading, but for profit considerations.
###############################################################################

def var(data, level=5):
    """
    """
    z = norm.ppf(level/100)
    return(-(data.mean() + z*data.std(ddof=0)))

def cvar(data, level=5):
    """
    """
    confidence_level = 1-(level/100)  # Set the desired confidence level
    sorted_prices = np.sort(data)
    num_samples = len(sorted_prices)
    cvar_index = int((1 - confidence_level) * num_samples)
    cvar = np.mean(sorted_prices[:cvar_index])
    return(cvar)

def drawdown(return_series):
    """
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({"Drawdown": drawdowns})

def sharpe(self,x):
    """
    """
    mu = x.mean()
    sigma = np.sqrt(x.var())
    return((mu-5) / sigma)

#def sortino(risk_free,degree_of_freedom,growth_rate,minimum):
    #v=np.sqrt(np.abs(scipy.integrate.quad(lambda g ((risk_freeg)**2)*scipy.stats.t.pdf(g, degree_of_freedom), risk_free, minimum)))
    #s=(growth_rate-risk_free)/v[0]

    #return s


###############################################################################
#                        statistical and quantitative indicators
#                        Backtest, Up-down indicator, trend indicator etc.
#                        These need a series of updates
###############################################################################

# unsure of functionality
def backtest(x, window = 21):
    weights = x.shift(1).rolling(window= window).std().dropna() / np.mean(x)
    return(x * weights.shift(1)).sum()  
    

def UD_ind(data):
    """
    """
    result = pd.DataFrame()
    result['returns'] = data
    
    lst = ["","","",""]

    for i in range(4,len(data)):

        st = ""

        if data[i-3] < 0:
            st = "D"
        elif data[i-3] >= 0:
            st = "U"

        if data[i-2] < 0:
            st = st + "D"
        elif data[i-2] >= 0:
            st = st + "U"

        if data[i-1] < 0:
            st = st +"D"
        elif data[i-1] >= 0:
            st = st + "U"

        lst.append(st)

    result['UD3indicator'] = lst

    lst = ["", "", "", "", ""]

    for i in range(5,len(data)):

        st = ""

        if data.iloc[i-5] < 0:
            st = "D"
        elif data.iloc[i-5] >= 0:
            st = "U"

        if data.iloc[i-4] < 0:
            st = st + "D"
        elif data.iloc[i-4] >= 0:
            st = st + "U"

        if data.iloc[i-3] < 0:
            st = st +"D"
        elif data.iloc[i-3] >= 0:
            st = st + "U"

        if data.iloc[i-2] < 0:
            st = st +"D"
        elif data.iloc[i-2] >= 0:
            st = st +"U"

        if data.iloc[i-1] < 0:
            st = st + "D"
        elif data.iloc[i-1] >= 0:
            st = st + "U"

        lst.append(st)

    result['UD5indicator'] = lst
    
    return(result)

def trend_ind(data):
    """
    """
    assert data.columns[3] == "Close"
    assert data.columns[7] == "velocity"

    result = data.copy()
    Trend = [None] * len(result)
    for i in range(21, len(result)):
        Trend[i] = tm.mk_test(result['Close'][i-21:i], 'full', window = 21)[0]
    
    Trend2 = [None] * len(result)
    for i in range(21, len(result)):
        t_ind = np.mean(result['velocity'][i-21:i]) * 100
        if t_ind < -0.5:
            Trend2[i] = "decreasing"
        elif t_ind > 0.5:
            Trend2[i] = "increasing"
        else:
            Trend2[i] = "no trend"     
    result = pd.DataFrame([Trend, Trend2]).T
    return(result)

def hidden_states(data):
    data = data.dropna()
    data.replace([np.inf, -np.inf, np.nan], 1, inplace=True)

    close = np.array(data.iloc[1:,3])
    vel = np.array(data.iloc[1:,7])
    acc = np.array(data.iloc[1:,8])
    vol = np.array(data.iloc[1:,4])
    vol_vel = np.array(data.iloc[1:,9])
    vol_acc = np.array(data.iloc[1:,10])


    RSI = np.array(data.iloc[1:,11])
    MA3 = np.array(data.iloc[1:,13])
    STD3 = np.array(data.iloc[1:,14])
    MA5 = np.array(data.iloc[1:,18])
    STD5 = np.array(data.iloc[1:,19])
    MA10 = np.array(data.iloc[1:,23])
    STD10 = np.array(data.iloc[1:,24])
    MA60 = np.array(data.iloc[1:,38])
    STD60 = np.array(data.iloc[1:,39])


    X = np.column_stack([close, vel, acc, vol, vol_vel, vol_acc, RSI, MA3, STD3, MA5, STD5, MA10, STD10, MA60, STD60])
    model = GaussianHMM(n_components=6, covariance_type="diag", n_iter=10000, random_state = 123).fit(X)
    hidden_states = model.predict(X)
    data = data.iloc[1:,:]
    data['States'] = hidden_states
    return(data)



