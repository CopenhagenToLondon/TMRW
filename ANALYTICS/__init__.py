"""
TMRW.ANALYTICS

Created on Mon Apr 15 22:44:02 2024

@author: Markb
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





class data_analytics:
    
    def __init__(self, data = None):
        
        try:
            
            if type(data) == pd.DataFrame:
                self.dataset = data

            elif isinstance(data, Simulations): # Hvis vi har simuleringer
                self.dataset = data.paths # Så tag paths

            #elif isinstance(data, Credit):

            #elif not isinstance(data, Simulations):
                #self.dataset = data

            else: 
                self.dataset = data
        
        #take into account that the Simulations class might be empty or none existant
        except NameError:
        
            self.dataset = data
        
        
        # take into account that datasets can be empty
        if len(self.dataset) <= 1:
                raise ValueError('This class needs a dataset to proceed')
        
        self.median()
        self.mean()
        self.variance()
        self.std()
        self.min = min(self.dataset) # skal rettes
        self.max = max(self.dataset) # skal rettes
        
    def median(self):
        
        if type(self.dataset) == pd.DataFrame:
            
            self.median = pd.DataFrame(columns = self.dataset.columns)
            self.median.loc['Median'] = None
            
            for i in range(len(self.dataset.columns)):
                
                lst = list(self.dataset[i])
                sortedLst = sorted(lst)
                lstLen = len(lst)
                index = (lstLen - 1) // 2

                if (lstLen % 2):
                    
                    self.median.iloc[0,i] = sortedLst[index]
                    
                else:
                    
                    self.median.iloc[0,i] = (sortedLst[index] + sortedLst[index + 1])/2.0
        
        elif type(self.dataset) == list:
            
            sortedLst = sorted(self.dataset)
            lstLen = len(self.dataset)
            index = (lstLen - 1) // 2

            if (lstLen % 2):
                
                self.median = sortedLst[index]
                
            else:
                
                self.median = (sortedLst[index] + sortedLst[index + 1])/2.0

    def mean(self):
       
        if type(self.dataset) == pd.DataFrame:
            
            self.mean = pd.DataFrame(columns = self.dataset.columns)
            
            self.mean.loc['Mean'] = None
            
            for i in range(len(self.dataset.columns)):
                
                self.mean.iloc[0,i] = np.float64(sum(self.dataset[i])/len(self.dataset))
        
        elif type(self.dataset) == list:
            
            self.mean = sum(self.dataset)/len(self.dataset)

        return(self.mean)
    
    
    def ma(data, t):

        if len(data) < t+1:
            raise ValueError("not enough data")

        mrt_df = []
        for i in range(len(data)):

            mean_reversion_times = np.zeros(t)

            for j in range(1,t):

                if i >= j:

                    mean_reversion_times[j] = np.mean(data[i-j:i])

            mrt_df.append(mean_reversion_times)

        mrt_df = pd.DataFrame(mrt_df, index = data.index)
        mrt_df = mrt_df.iloc[1:len(mrt_df), 1:len(mrt_df.columns)]
        return(mrt_df)
    
    
    
        
    def variance(self, typ = "population"):
        
        self.typ = typ
       
        if type(self.dataset) == pd.DataFrame:
            
            self.var = pd.DataFrame(columns = self.dataset.columns)
            self.var.loc['Variance'] = None
            
            for i in range(len(self.dataset.columns)):
                self.var.iloc[0,i] = 0
                
                for j in range(len(self.dataset)):
                    
                    self.var.iloc[0,i] = self.var.iloc[0,i] + (self.dataset.iloc[j,i] - self.mean.iloc[0,i])**2
                
                if typ == "population":
                    
                    self.var.iloc[0,i] = self.var.iloc[0,i] / len(self.dataset)
                    
                elif typ == "sample":
                    
                    self.var.iloc[0,i] = self.var.iloc[0,i] / (len(self.dataset) - 1)
            
        
        elif type(self.dataset) == list:
            
            self.var = None
            
            for i in range(len(self.dataset)):
                
                self.var = self.var + (self.dataset[i]-self.mean)**2
            
            if typ == "population":
                
                self.var = self.var / len(self.dataset)
                
            elif typ == "sample":
                
                self.var = self.var / (len(self.dataset) - 1)
        
        return(self.var)
        
    def std(self, typ = "population"):
        
        self.typ = typ
        self.variance(self.typ)
        
        self.stdev = self.var ** (1/2)
        self.stdev.index = ['Standard Deviation']
        return(self.stdev)
    
    def stdev(data, t):

        if len(data) < t+1:
            raise ValueError("not enough data")

        mrt_df = []
        for i in range(len(data)):

            mean_reversion_times = np.zeros(t)

            for j in range(1,t):

                if i >= j:

                    mean_reversion_times[j] = np.std(data[i-j:i])

            mrt_df.append(mean_reversion_times)

        mrt_df = pd.DataFrame(mrt_df, index = data.index)
        mrt_df = mrt_df.iloc[1:len(mrt_df), 1:len(mrt_df.columns)]
        return(mrt_df)
    
    
    def slope(self, length = 1):
        
        if type(self.dataset) == pd.DataFrame:
            
            self.slope = pd.DataFrame(None,index = self.dataset.index, columns = self.dataset.columns)
            
            for i in range(len(self.dataset.columns)):
                for j in range(0+length,len(self.dataset)):
                    
                    self.slope.iloc[j,i] = (self.dataset.iloc[j,i] - self.dataset.iloc[j-length,i]) / (self.dataset.index[j] - self.dataset.index[j-length])

    def stationarity(self):
        result = adfuller(self.dataset[0])
        res = "Stationary"
        # = pd.DataFrame(result[4].items())[1]
        if result[1] > 0.1:
            res = "Non-stationary"
        return res
     
    # estimate best probability distribution
    
    # estimate parameters
    
    # regression models
    
    #empirical distribution
     
    def empirical(self, data, bins = 10):
    
        bine = np.linspace(min(data), max(data), bins)
        counts = []
        
        for i in range(0,len(bine)):
    
            bine[i] = round(bine[i],3)
            count = 0
    
            for j in range(len(data)):
    
                if data.iloc[j] >= bine[i-1] and data.iloc[j] < bine[i]:
    
                    count = count + 1
    
            counts.append(count/len(data))
    
        counts_df = pd.DataFrame(counts, index = bine)
        return(counts_df)
    
    
    
    
    
    
class analysis:
    
    def __init__(self):
        print("hello world")
        today = date.today() 
        self.today = datetime(today.year,today.month,today.day) #today
        self.one = datetime(today.year-1,today.month,today.day) #one year ago
        self.three = datetime(today.year-3,today.month,today.day) #three years ago
        self.five = datetime(today.year-5,today.month,today.day) #five years ago
        self.ten = datetime(today.year-10,today.month,today.day) #ten years ago
        self.twenty = datetime(today.year-20,today.month,today.day) #twenty years ago
        
    def updown(data):
     
         u_d_count = []
         
         for i in range(len(data)):
             
             if data.iloc[i] < 0:
                 
                 u_d_count.append(-1)
             
             elif data.iloc[i] > 0:
                 
                 u_d_count.append(1)
                 
             else:
                 
                 u_d_count.append(0)
                 
         return(u_d_count) 
   
    def updown_prob(data, k = 1):
        
        prob = pd.DataFrame(0,index = ['up', 'down'], columns = ['up', 'down'])
        
        # columns are "future"
        # rows are "past"
        
        for i in range(len(data)):
        
            if data[i] > 0 and data[i-k] > 0:
        
                prob['up']['up'] = prob['up']['up'] + 1
        
            elif data[i] > 0 and data[i-k] < 0:
        
                prob['up']['down'] = prob['up']['down'] + 1
        
            elif data[i] < 0 and data[i-k] < 0:
        
                prob['down']['down'] = prob['down']['down'] + 1
        
            elif data[i] < 0 and data[i-k] > 0:
        
                prob['down']['up'] = prob['down']['up'] + 1
        
        prob = prob / len(data)
        return(prob)
   
    #markov chain probabilities

    def counts(data):
        ind = ['U','UU','UUU','UUD','UD','UDU','UDD','D','DU','DUU','DUD','DD','DDU','DDD'] #index
        
        counts = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        
        for i in range(1, len(data)):
    
            # if [-1] is up. Add +1 to counts[0]
            if data[i-1] > 0:   
                counts[0] = counts[0] + 1
    
            # UU
            if data[i-1] > 0 and data[i-2] > 0:
                counts[1] = counts[1] + 1
    
            # UUU
            if data[i-1] > 0 and data[i-2] > 0 and data[i-3] > 0:
                counts[2] = counts[2] + 1
    
            # UUD
            if data[i-1] > 0 and data[i-2] > 0 and data[i-3] < 0:
                counts[3] = counts[3] + 1
    
            # UD
            if data[i-1] > 0 and data[i-2] < 0:
                counts[4] = counts[4] + 1
    
            # UDU
            if data[i-1] > 0 and data[i-2] < 0 and data[i-3] < 0:
                counts[5] = counts[5] + 1
    
            # UDD
            if data[i-1] > 0 and data[i-2] < 0 and data[i-3] > 0:
                counts[6] = counts[6] + 1
    
    
            # if [-1] is down. Add +1 to counts[7]
            if data[i-1] < 0:   
                counts[7] = counts[7] + 1
    
            # DU
            if data[i-1] < 0 and data[i-2] > 0:
                counts[8] = counts[8] + 1
    
            # DUU
            if data[i-1] < 0 and data[i-2] > 0 and data[i-3] > 0:
                counts[9] = counts[9] + 1
    
            # DUD
            if data[i-1] < 0 and data[i-2] > 0 and data[i-3] < 0:
                counts[10] = counts[10] + 1
    
            # DD
            if data[i-1] < 0 and data[i-2] < 0:
                counts[11] = counts[11] + 1
    
            # DDU
            if data[i-1] < 0 and data[i-2] < 0 and data[i-3] < 0:
                counts[12] = counts[12] + 1
    
            # DDD
            if data[i-1] < 0 and data[i-2] < 0 and data[i-3] > 0:
                counts[13] = counts[13] + 1
            
        count = pd.DataFrame(counts, index = ind, columns = ['count']) # count table
        return(count)
    
    def probs(data):
        ind = ['U','UU','UUU','UUD','UD','UDU','UDD','D','DU','DUU','DUD','DD','DDU','DDD'] #index
        
        up = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        down = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        
        for i in range(1, len(data)):
            
            if data[i] > 0:
    
                # if [-1] is up. Add +1 to counts[0]
                if data[i-1] > 0:   
                    up[0] = up[0] + 1
    
                # UU
                if data[i-1] > 0 and data[i-2] > 0:
                    up[1] = up[1] + 1
    
                # UUU
                if data[i-1] > 0 and data[i-2] > 0 and data[i-3] > 0:
                    up[2] = up[2] + 1
    
                # UUD
                if data[i-1] > 0 and data[i-2] > 0 and data[i-3] < 0:
                    up[3] = up[3] + 1
    
                # UD
                if data[i-1] > 0 and data[i-2] < 0:
                    up[4] = up[4] + 1
    
                # UDU
                if data[i-1] > 0 and data[i-2] < 0 and data[i-3] < 0:
                    up[5] = up[5] + 1
    
                # UDD
                if data[i-1] > 0 and data[i-2] < 0 and data[i-3] > 0:
                    up[6] = up[6] + 1
    
    
                # if [-1] is down. Add +1 to counts[7]
                if data[i-1] < 0:   
                    up[7] = up[7] + 1
    
                # DU
                if data[i-1] < 0 and data[i-2] > 0:
                    up[8] = up[8] + 1
    
                # DUU
                if data[i-1] < 0 and data[i-2] > 0 and data[i-3] > 0:
                    up[9] = up[9] + 1
    
                # DUD
                if data[i-1] < 0 and data[i-2] > 0 and data[i-3] < 0:
                    up[10] = up[10] + 1
    
                # DD
                if data[i-1] < 0 and data[i-2] < 0:
                    up[11] = up[11] + 1
    
                # DDU
                if data[i-1] < 0 and data[i-2] < 0 and data[i-3] < 0:
                    up[12] = up[12] + 1
    
                # DDD
                if data[i-1] < 0 and data[i-2] < 0 and data[i-3] > 0:
                    up[13] = up[13] + 1
            
            elif data[i] < 0:
                
                            # if [-1] is up. Add +1 to counts[0]
                if data[i-1] > 0:   
                    down[0] = down[0] + 1
    
                # UU
                if data[i-1] > 0 and data[i-2] > 0:
                    down[1] = down[1] + 1
    
                # UUU
                if data[i-1] > 0 and data[i-2] > 0 and data[i-3] > 0:
                    down[2] = down[2] + 1
    
                # UUD
                if data[i-1] > 0 and data[i-2] > 0 and data[i-3] < 0:
                    down[3] = down[3] + 1
    
                # UD
                if data[i-1] > 0 and data[i-2] < 0:
                    down[4] = down[4] + 1
    
                # UDU
                if data[i-1] > 0 and data[i-2] < 0 and data[i-3] < 0:
                    down[5] = down[5] + 1
    
                # UDD
                if data[i-1] > 0 and data[i-2] < 0 and data[i-3] > 0:
                    down[6] = down[6] + 1
    
    
                # if [-1] is down. Add +1 to counts[7]
                if data[i-1] < 0:   
                    down[7] = down[7] + 1
    
                # DU
                if data[i-1] < 0 and data[i-2] > 0:
                    down[8] = down[8] + 1
    
                # DUU
                if data[i-1] < 0 and data[i-2] > 0 and data[i-3] > 0:
                    down[9] = down[9] + 1
    
                # DUD
                if data[i-1] < 0 and data[i-2] > 0 and data[i-3] < 0:
                    down[10] = down[10] + 1
    
                # DD
                if data[i-1] < 0 and data[i-2] < 0:
                    down[11] = down[11] + 1
    
                # DDU
                if data[i-1] < 0 and data[i-2] < 0 and data[i-3] < 0:
                    down[12] = down[12] + 1
    
                # DDD
                if data[i-1] < 0 and data[i-2] < 0 and data[i-3] > 0:
                    down[13] = down[13] + 1
            
        count = pd.DataFrame(up, index = ind, columns = ['up']) # count table
        count['down'] = down
        return(count)
   
    
        def moving_averages(data, t):
            
            if len(data) < t+1:
                raise ValueError("not enough data")
                
            mrt_df = []
            for i in range(len(data)):
                
                mean_reversion_times = np.zeros(t)
                
                for j in range(1,t):
                
                    if i >= j:
                    
                        mean_reversion_times[j] = np.mean(data[i-j:i])
                
                mrt_df.append(mean_reversion_times)
        
            mrt_df = pd.DataFrame(mrt_df, index = data.index)
            mrt_df = mrt_df.iloc[1:len(mrt_df), 1:len(mrt_df.columns)]
            #lst[0] = list(x)
            return(mrt_df)
        
        def mean_sizes(data, t):
            
            sizes = moving_averages(data, t)
            for i in range(len(sizes)):
                
                for j in range(len(sizes.columns)):
                    
                    sizes.iloc[i,j] = data.iloc[i] - sizes.iloc[i,j] 
                    
            return(sizes)
        
        def mean_size_dist(data, t):
            
            ms = mean_sizes(data,t)
            msd = pd.DataFrame(0, index = ['upper+2s', 'upper-m', 'upper-2s','lower+2s','lower-m','lower-2s'], columns = ms.columns)
            for ti in range(1, len(ms.columns)):
                
                msp = ms[ti+1][ms[ti+1] > 0]
                msn = ms[ti+1][ms[ti+1] < 0]
                
                msd[ti+1]['upper+2s'] = np.mean(msp) + 2 * np.std(msp)
                msd[ti+1]['upper-m'] = np.mean(msp)
                msd[ti+1]['upper-2s'] = min(msp)
                
                msd[ti+1]['lower+2s'] = max(msn)
                msd[ti+1]['lower-m'] = np.mean(msn)
                msd[ti+1]['lower-2s'] = np.mean(msn) - 2 * np.std(msn)
                
            return(msd)
                
        def mean_times(data, t):
            ms = mean_sizes(data, t)
            mrt = pd.DataFrame(index = ["mrt"], columns = ms.columns)
            for i in range(len(ms.columns)):
                n = 0
                for j in range(len(ms)):
                    if ms.iloc[j,i] > 0 and ms.iloc[j-1,i] < 0:
                        n = n + 1
                    elif ms.iloc[j,i] < 0 and ms.iloc[j-1,i] > 0:
                        n = n + 1
                    else:
                        n = n
                mrt[i+1]['mrt'] = len(ms) / (n+1)
            return(mrt)      
        
        def month_probabilities(self,Tickers):
            p = pd.DataFrame(0.0, index = ['p'], columns = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
            for j in range(1,13):
                russel_prob = 0.0
            
                for tick in Tickers:
            
                    Aktie = sdata(tick, '2000-01-01', self.today)
                    aar = list(Aktie.index)
                    for i in range(len(aar)):
                        aar[i] = str(aar[i])[0:4]
            
                    aar = list(dict.fromkeys(aar))
            
                    r_prob = 0.0
            
                    if len(str(j)) == 2 and (j-1) != 9:
                        l_s = "-"+str(j) + "-01"
                        u_s = "-"+str(j-1) + "-01"
                    elif len(str(j)) == 2 and (j-1) == 9:
                        l_s = "-"+str(j) + "-01"
                        u_s = "-0"+str(j-1) + "-01"
                    elif len(str(j)) != 2 and (j-1) != 0:
                        l_s = "-0"+str(j)+"-01"
                        u_s = "-0"+str(j-1)+"-01"
                    elif len(str(j)) != 2 and (j-1) == 0:
                        l_s = "-01-31"
                        u_s = "-01-01"
            
            
            
            
                    for i in range(1,len(aar)-1):
            
                        june_july = Aktie[Aktie.index < str(aar[i]) + l_s]
                        june_july = june_july[june_july.index >= str(aar[i]) + u_s]
                        if june_july.iloc[0] < june_july.iloc[len(june_july)-1]:
                            r_prob = r_prob + 1.0
            
                    russel_prob = russel_prob + r_prob / len(aar)
            
                p.iloc[0,j-1] = np.float64(russel_prob / len(Tickers))
            
            return(p)

        def extremum_probabilities(self,Tickers):
            probabilities = pd.DataFrame(index = ['min', 'max'], columns = ['7dage', '1måned'])
            probabilities.iloc[0,0] = 0.0
            probabilities.iloc[1,0] = 0.0
            probabilities.iloc[0,1] = 0.0
            probabilities.iloc[1,1] = 0.0
        
            for tick in Tickers:
        
                Aktie = sdata(tick, '2014-01-01', self.today)
                aar = list(Aktie.index)
                for i in range(len(aar)):
                    aar[i] = str(aar[i])[0:4]
        
                aar = list(dict.fromkeys(aar))
        
                probs = pd.DataFrame(0, index = ['min', 'max'], columns = ['7dage', '1måned'])
        
                for i in range(len(aar)-1):
        
                    temp_aktie = Aktie[Aktie.index < str(aar[i+1]) + "-01-30"]
                    temp_aktie = temp_aktie[temp_aktie.index >= str(aar[i]) + "-01-01"]
        
                    for j in range(len(temp_aktie)):
        
                        if str(temp_aktie.index[j]) < str(aar[i])+"12-29":
                            if temp_aktie.iloc[j] == min(temp_aktie) and temp_aktie.iloc[j + 6] > 1.01 * temp_aktie.iloc[j]:
                                probs.iloc[0,0] = probs.iloc[0,0] + 1.0
        
                            if temp_aktie.iloc[j] == min(temp_aktie) and temp_aktie.iloc[j + 19] > 1.05 * temp_aktie.iloc[j]:
                                probs.iloc[0,1] = probs.iloc[0,1] + 1.0
        
                            if temp_aktie.iloc[j] == max(temp_aktie) and temp_aktie.iloc[j + 6] < 0.99 * temp_aktie.iloc[j]:
                                probs.iloc[1,0] = probs.iloc[1,0] + 1.0
        
                            if temp_aktie.iloc[j] == max(temp_aktie) and temp_aktie.iloc[j + 19] < 0.95 * temp_aktie.iloc[j]:
                                probs.iloc[1,1] = probs.iloc[1,1] + 1.0
        
                probs = probs / len(aar)
        
                probabilities.iloc[0,0] = probabilities.iloc[0,0] + probs.iloc[0,0]
                probabilities.iloc[1,0] = probabilities.iloc[1,0] + probs.iloc[1,0]
                probabilities.iloc[0,1] = probabilities.iloc[0,1] + probs.iloc[0,1]
                probabilities.iloc[1,1] = probabilities.iloc[1,1] + probs.iloc[1,1]
        
            probabilities = probabilities / len(Tickers)         
            return(probabilities)
    
    
    
    
    
    def wealth_plot(x):
        titel = "Wealth plot " + str(x.columns[0])
        plt.figure(figsize=(18, 8))
        plt.plot((x + 1).cumprod())
        plt.grid()
        plt.title(titel)
        plt.xlabel("Date")
        plt.ylabel("Wealth")
        plt.show()

    def frequencies(data, bins, wp = False):
        #binning strategien skal opdateres på et senere tidspunkt
        if type(data) != pd.DataFrame:
            data = pd.DataFrame(data)
        
        m1 = min(data.iloc[:,0])
        m2 = max(data.iloc[:,0])    
        res1 = []
        res2 = []
        xlist = []
            
        for i in range(bins):
            k = 0

            inter1 = m1 + (i/bins) * (m2 - m1)
            inter2 = m1 + ((i+1) /bins) * (m2 -m1)
            xlist.append(round(inter2,5))

            for j in range(len(data)):
                if float(data.iloc[j,0]) > float(inter1) and float(data.iloc[j,0]) < float(inter2) :
                    k = k + 1

            res1.append(round(int(k),5))
            res2.append(round(k/len(data),5))
        
        
        res = pd.DataFrame([res1,res2],index = ['Frequencies', 'Percentages'], columns = xlist)
        lst = np.linspace(m1, m2, bins)

        if wp == True:
            y_pos = np.arange(len(res.columns))
            plt.bar(y_pos, res.iloc[0], color ='navy', width = 1)
            plt.xticks(y_pos,res.columns)
            plt.ylabel("Frequencies")
            plt.xlabel("Percentage returns")
            plt.title("Frequencies")
            plt.plot(y_pos, res.iloc[0], color = 'maroon')
            plt.xticks(y_pos,res.columns)
            plt.tick_params(axis='both', which='major', labelsize=7)
        return res

    def sstat(data, txt = "Stat"):
        m = pd.DataFrame([0])
        for n in range(0, 4):
            s = 0
            if n == 0:
                for j in range(len(data)):
                    if type(data) == pd.Series:
                        s = s + data[j]
                    elif type(data) == pd.DataFrame:
                        s = s + data.iloc[j,0]
                s = s/ len(data)

            elif n > 0:
                for j in range(len(data)):
                    if type(data) == pd.Series:
                        s = s + (data[j] - m[0][0])**(n+1)
                    elif type(data) == pd.DataFrame:
                        s = s + (data.iloc[j,0] - m[0][0])**(n+1)
                s = s / len(data)

            if n == 0 or n == 1:
                m[n] = s
            elif n>1:
                m[n] = s / (np.sqrt(m[1][0])**(n+1))
            
            if type(data) == pd.DataFrame:
                p = [data.describe().squeeze()[3],data.describe().squeeze()[4],data.describe().squeeze()[5],data.describe().squeeze()[6], data.describe().squeeze()[7]]
            else:
                p = [data.describe()[3],data.describe()[4],data.describe()[5],data.describe()[6],data.describe()[7]]

        mean = {'min':[p[0]],'25':[p[1]],'50':[p[2]],'75':[p[3]],'max':[p[4]],'Mean':[m[0][0]],'STD':[np.sqrt(m[1][0])], 'Variance':[m[1][0]], 'Skew':[m[2][0]], 'Kurtosis':[m[3][0]]}
        m = pd.DataFrame(mean, index = [str(txt)])
        return m

    def seasonality(S):
        lst = [0,0,0,0,0,0,0,0,0,0,0,0]
        liste = []
        j = 0
        k = 0
        f = 0
        for i in range(len(S)):

            if int(str(S.index[i])[0:4]) > int(str(S.index[i-1])[0:4]):
                liste.append(lst)
                j = 0
                k = 0
                lst = [0,0,0,0,0,0,0,0,0,0,0,0]

            if int(str(S.index[i])[5:7]) > int(str(S.index[i-1])[5:7]) :
                j = 0
                k = 0

            j = j + 1   
            k = k + S[i]
            f = k/j

            lst[int(str(S.index[i])[5:7])-1] = f

        SS = pd.DataFrame(liste, columns = ['JAN', 'FEB', 'MAR', 'APR', 'MAJ', 'JUN', 'JUL', 'AUG', 'SEP', 'OKT', 'NOV', 'DEC'])
        return SS

    def seasonmax(x):

        lstmin = [0,0,0,0,0,0,0,0,0,0,0,0]
        lstmax = [0,0,0,0,0,0,0,0,0,0,0,0]
        for aar in range(len(x)): 

            for j in range(len(x.columns)):
                maan = x.columns[j]
                if x[maan][aar] == 0:
                    break

                if x[maan][aar] == min(x.loc[aar]):
                    lstmin[j] = lstmin[j] + 1

                elif x[maan][aar] == max(x.loc[aar]):
                    lstmax[j] = lstmax[j] + 1

        LL = pd.DataFrame([lstmin,lstmax], columns = ['JAN', 'FEB', 'MAR', 'APR', 'MAJ', 'JUN', 'JUL', 'AUG', 'SEP', 'OKT', 'NOV', 'DEC'], index = ['Min', 'Max'])
        return LL




    ########################################################################################################
    # Empirical distribution function
    ########################################################################################################
    def ed(data, bins = 10):

        bine = np.linspace(min(data), max(data), bins)
        counts = []

        for i in range(0,len(bine)):

            bine[i] = round(bine[i],3)
            count = 0

            for j in range(len(data)):

                if data.iloc[j] >= bine[i-1] and data.iloc[j] < bine[i]:

                    count = count + 1

            counts.append(count/len(data))

        counts_df = pd.DataFrame(counts, index = bine)
        return(counts_df)



        
    def indicator(self, returns):
        self.indic = returns.copy()
        for i in range(len(self.indic)):
            if self.indic.iloc[i] < 0:
                self.indic.iloc[i] = -1
            elif self.indic.iloc[i] > 0:
                self.indic.iloc[i] = 1
            else:
                self.indic.iloc[i] = 0
        return(self.indic)  

    def n_d_indicator(self, indic, t):
        
        indic_d = indic.copy()
        
        for i in range(t, len(indic)):
            
            indic_d.iloc[i] = sum(indic[i-t:i])
            
        return(indic_d)
    
    
    
            
def updown(data):
    
    u_d_count = []
    
    for i in range(len(data)):
        
        if data.iloc[i] < 0:
            
            u_d_count.append(-1)
        
        elif data.iloc[i] > 0:
            
            u_d_count.append(1)
            
        else:
            
            u_d_count.append(0)
            
    return(u_d_count)
    
def updown_prob(data, k = 1):

    prob = pd.DataFrame(0,index = ['up', 'down'], columns = ['up', 'down'])

    # columns are "future"
    # rows are "past"

    for i in range(len(data)):

        if data[i] > 0 and data[i-k] > 0:

            prob['up']['up'] = prob['up']['up'] + 1

        elif data[i] > 0 and data[i-k] < 0:

            prob['up']['down'] = prob['up']['down'] + 1

        elif data[i] < 0 and data[i-k] < 0:

            prob['down']['down'] = prob['down']['down'] + 1

        elif data[i] < 0 and data[i-k] > 0:

            prob['down']['up'] = prob['down']['up'] + 1

    prob = prob / len(data)
    return(prob)      



#markov chain probabilities

def counts(data):
    ind = ['U','UU','UUU','UUD','UD','UDU','UDD','D','DU','DUU','DUD','DD','DDU','DDD'] #index
    
    counts = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    
    for i in range(1, len(data)):

        # if [-1] is up. Add +1 to counts[0]
        if data[i-1] > 0:   
            counts[0] = counts[0] + 1

        # UU
        if data[i-1] > 0 and data[i-2] > 0:
            counts[1] = counts[1] + 1

        # UUU
        if data[i-1] > 0 and data[i-2] > 0 and data[i-3] > 0:
            counts[2] = counts[2] + 1

        # UUD
        if data[i-1] > 0 and data[i-2] > 0 and data[i-3] < 0:
            counts[3] = counts[3] + 1

        # UD
        if data[i-1] > 0 and data[i-2] < 0:
            counts[4] = counts[4] + 1

        # UDU
        if data[i-1] > 0 and data[i-2] < 0 and data[i-3] < 0:
            counts[5] = counts[5] + 1

        # UDD
        if data[i-1] > 0 and data[i-2] < 0 and data[i-3] > 0:
            counts[6] = counts[6] + 1


        # if [-1] is down. Add +1 to counts[7]
        if data[i-1] < 0:   
            counts[7] = counts[7] + 1

        # DU
        if data[i-1] < 0 and data[i-2] > 0:
            counts[8] = counts[8] + 1

        # DUU
        if data[i-1] < 0 and data[i-2] > 0 and data[i-3] > 0:
            counts[9] = counts[9] + 1

        # DUD
        if data[i-1] < 0 and data[i-2] > 0 and data[i-3] < 0:
            counts[10] = counts[10] + 1

        # DD
        if data[i-1] < 0 and data[i-2] < 0:
            counts[11] = counts[11] + 1

        # DDU
        if data[i-1] < 0 and data[i-2] < 0 and data[i-3] < 0:
            counts[12] = counts[12] + 1

        # DDD
        if data[i-1] < 0 and data[i-2] < 0 and data[i-3] > 0:
            counts[13] = counts[13] + 1
        
    count = pd.DataFrame(counts, index = ind, columns = ['count']) # count table
    return(count)




def probs(data):
    ind = ['U','UU','UUU','UUD','UD','UDU','UDD','D','DU','DUU','DUD','DD','DDU','DDD'] #index
    
    up = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    down = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    
    for i in range(1, len(data)):
        
        if data[i] > 0:

            # if [-1] is up. Add +1 to counts[0]
            if data[i-1] > 0:   
                up[0] = up[0] + 1

            # UU
            if data[i-1] > 0 and data[i-2] > 0:
                up[1] = up[1] + 1

            # UUU
            if data[i-1] > 0 and data[i-2] > 0 and data[i-3] > 0:
                up[2] = up[2] + 1

            # UUD
            if data[i-1] > 0 and data[i-2] > 0 and data[i-3] < 0:
                up[3] = up[3] + 1

            # UD
            if data[i-1] > 0 and data[i-2] < 0:
                up[4] = up[4] + 1

            # UDU
            if data[i-1] > 0 and data[i-2] < 0 and data[i-3] < 0:
                up[5] = up[5] + 1

            # UDD
            if data[i-1] > 0 and data[i-2] < 0 and data[i-3] > 0:
                up[6] = up[6] + 1


            # if [-1] is down. Add +1 to counts[7]
            if data[i-1] < 0:   
                up[7] = up[7] + 1

            # DU
            if data[i-1] < 0 and data[i-2] > 0:
                up[8] = up[8] + 1

            # DUU
            if data[i-1] < 0 and data[i-2] > 0 and data[i-3] > 0:
                up[9] = up[9] + 1

            # DUD
            if data[i-1] < 0 and data[i-2] > 0 and data[i-3] < 0:
                up[10] = up[10] + 1

            # DD
            if data[i-1] < 0 and data[i-2] < 0:
                up[11] = up[11] + 1

            # DDU
            if data[i-1] < 0 and data[i-2] < 0 and data[i-3] < 0:
                up[12] = up[12] + 1

            # DDD
            if data[i-1] < 0 and data[i-2] < 0 and data[i-3] > 0:
                up[13] = up[13] + 1
        
        elif data[i] < 0:
            
                        # if [-1] is up. Add +1 to counts[0]
            if data[i-1] > 0:   
                down[0] = down[0] + 1

            # UU
            if data[i-1] > 0 and data[i-2] > 0:
                down[1] = down[1] + 1

            # UUU
            if data[i-1] > 0 and data[i-2] > 0 and data[i-3] > 0:
                down[2] = down[2] + 1

            # UUD
            if data[i-1] > 0 and data[i-2] > 0 and data[i-3] < 0:
                down[3] = down[3] + 1

            # UD
            if data[i-1] > 0 and data[i-2] < 0:
                down[4] = down[4] + 1

            # UDU
            if data[i-1] > 0 and data[i-2] < 0 and data[i-3] < 0:
                down[5] = down[5] + 1

            # UDD
            if data[i-1] > 0 and data[i-2] < 0 and data[i-3] > 0:
                down[6] = down[6] + 1


            # if [-1] is down. Add +1 to counts[7]
            if data[i-1] < 0:   
                down[7] = down[7] + 1

            # DU
            if data[i-1] < 0 and data[i-2] > 0:
                down[8] = down[8] + 1

            # DUU
            if data[i-1] < 0 and data[i-2] > 0 and data[i-3] > 0:
                down[9] = down[9] + 1

            # DUD
            if data[i-1] < 0 and data[i-2] > 0 and data[i-3] < 0:
                down[10] = down[10] + 1

            # DD
            if data[i-1] < 0 and data[i-2] < 0:
                down[11] = down[11] + 1

            # DDU
            if data[i-1] < 0 and data[i-2] < 0 and data[i-3] < 0:
                down[12] = down[12] + 1

            # DDD
            if data[i-1] < 0 and data[i-2] < 0 and data[i-3] > 0:
                down[13] = down[13] + 1
        
    count = pd.DataFrame(up, index = ind, columns = ['up']) # count table
    count['down'] = down
    return(count)
        
    







def moving_averages(data, t):
    
    if len(data) < t+1:
        raise ValueError("not enough data")
        
    mrt_df = []
    for i in range(len(data)):
        
        mean_reversion_times = np.zeros(t)
        
        for j in range(1,t):
        
            if i >= j:
            
                mean_reversion_times[j] = np.mean(data[i-j:i])
        
        mrt_df.append(mean_reversion_times)

    mrt_df = pd.DataFrame(mrt_df, index = data.index)
    mrt_df = mrt_df.iloc[1:len(mrt_df), 1:len(mrt_df.columns)]
    #lst[0] = list(x)
    return(mrt_df)

def mean_sizes(data, t):
    
    sizes = moving_averages(data, t)
    for i in range(len(sizes)):
        
        for j in range(len(sizes.columns)):
            
            sizes.iloc[i,j] = data.iloc[i] - sizes.iloc[i,j] 
            
    return(sizes)

def mean_size_dist(data, t):
    
    ms = mean_sizes(data,t)
    msd = pd.DataFrame(0, index = ['upper+2s', 'upper-m', 'upper-2s','lower+2s','lower-m','lower-2s'], columns = ms.columns)
    for ti in range(1, len(ms.columns)):
        
        msp = ms[ti+1][ms[ti+1] > 0]
        msn = ms[ti+1][ms[ti+1] < 0]
        
        msd[ti+1]['upper+2s'] = np.mean(msp) + 2 * np.std(msp)
        msd[ti+1]['upper-m'] = np.mean(msp)
        msd[ti+1]['upper-2s'] = min(msp)
        
        msd[ti+1]['lower+2s'] = max(msn)
        msd[ti+1]['lower-m'] = np.mean(msn)
        msd[ti+1]['lower-2s'] = np.mean(msn) - 2 * np.std(msn)
        
    return(msd)
        
def mean_times(data, t):
    ms = mean_sizes(data, t)
    mrt = pd.DataFrame(index = ["mrt"], columns = ms.columns)
    for i in range(len(ms.columns)):
        n = 0
        for j in range(len(ms)):
            if ms.iloc[j,i] > 0 and ms.iloc[j-1,i] < 0:
                n = n + 1
            elif ms.iloc[j,i] < 0 and ms.iloc[j-1,i] > 0:
                n = n + 1
            else:
                n = n
        mrt[i+1]['mrt'] = len(ms) / (n+1)
    return(mrt)      




def month_probabilities(Tickers):
    p = pd.DataFrame(0.0, index = ['p'], columns = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
    for j in range(1,13):
        russel_prob = 0.0

        for tick in Tickers:

            Aktie = tf.sdata(tick, '2000-01-01', today)
            aar = list(Aktie.index)
            for i in range(len(aar)):
                aar[i] = str(aar[i])[0:4]

            aar = list(dict.fromkeys(aar))

            r_prob = 0.0

            if len(str(j)) == 2 and (j-1) != 9:
                l_s = "-"+str(j) + "-01"
                u_s = "-"+str(j-1) + "-01"
            elif len(str(j)) == 2 and (j-1) == 9:
                l_s = "-"+str(j) + "-01"
                u_s = "-0"+str(j-1) + "-01"
            elif len(str(j)) != 2 and (j-1) != 0:
                l_s = "-0"+str(j)+"-01"
                u_s = "-0"+str(j-1)+"-01"
            elif len(str(j)) != 2 and (j-1) == 0:
                l_s = "-01-31"
                u_s = "-01-01"




            for i in range(1,len(aar)-1):

                june_july = Aktie[Aktie.index < str(aar[i]) + l_s]
                june_july = june_july[june_july.index >= str(aar[i]) + u_s]
                if june_july.iloc[0] < june_july.iloc[len(june_july)-1]:
                    r_prob = r_prob + 1.0

            russel_prob = russel_prob + r_prob / len(aar)

        p.iloc[0,j-1] = np.float64(russel_prob / len(Tickers))

    return(p)

def extremum_probabilities(Tickers):
    probabilities = pd.DataFrame(index = ['min', 'max'], columns = ['7dage', '1måned'])
    probabilities.iloc[0,0] = 0.0
    probabilities.iloc[1,0] = 0.0
    probabilities.iloc[0,1] = 0.0
    probabilities.iloc[1,1] = 0.0

    for tick in Tickers:

        Aktie = tf.sdata(tick, '2014-01-01', today)
        aar = list(Aktie.index)
        for i in range(len(aar)):
            aar[i] = str(aar[i])[0:4]

        aar = list(dict.fromkeys(aar))

        probs = pd.DataFrame(0, index = ['min', 'max'], columns = ['7dage', '1måned'])

        for i in range(len(aar)-1):

            temp_aktie = Aktie[Aktie.index < str(aar[i+1]) + "-01-30"]
            temp_aktie = temp_aktie[temp_aktie.index >= str(aar[i]) + "-01-01"]

            for j in range(len(temp_aktie)):

                if str(temp_aktie.index[j]) < str(aar[i])+"12-29":
                    if temp_aktie.iloc[j] == min(temp_aktie) and temp_aktie.iloc[j + 6] > 1.01 * temp_aktie.iloc[j]:
                        probs.iloc[0,0] = probs.iloc[0,0] + 1.0

                    if temp_aktie.iloc[j] == min(temp_aktie) and temp_aktie.iloc[j + 19] > 1.05 * temp_aktie.iloc[j]:
                        probs.iloc[0,1] = probs.iloc[0,1] + 1.0

                    if temp_aktie.iloc[j] == max(temp_aktie) and temp_aktie.iloc[j + 6] < 0.99 * temp_aktie.iloc[j]:
                        probs.iloc[1,0] = probs.iloc[1,0] + 1.0

                    if temp_aktie.iloc[j] == max(temp_aktie) and temp_aktie.iloc[j + 19] < 0.95 * temp_aktie.iloc[j]:
                        probs.iloc[1,1] = probs.iloc[1,1] + 1.0

        probs = probs / len(aar)

        probabilities.iloc[0,0] = probabilities.iloc[0,0] + probs.iloc[0,0]
        probabilities.iloc[1,0] = probabilities.iloc[1,0] + probs.iloc[1,0]
        probabilities.iloc[0,1] = probabilities.iloc[0,1] + probs.iloc[0,1]
        probabilities.iloc[1,1] = probabilities.iloc[1,1] + probs.iloc[1,1]

    probabilities = probabilities / len(Tickers)         
    return(probabilities)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
