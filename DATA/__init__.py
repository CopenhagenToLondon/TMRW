"""
TMRW.DATA

Created on Mon Apr 15 22:44:02 2024

@author: Markb
"""


import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta, timezone 
import sqlalchemy as sa
from hmmlearn.hmm import GaussianHMM

import TMRW.FINANCE as tf


def data(x, y, z):
    # x is a stock symbol
    # y is a start date
    # z is an end date
    # kan bruges på commodities, currencies og aktier
    
    #MSFT er en aktier der kan bruges
    #USDEUR=X er en currency der kan bruges
    #GC=F er guld priser
    
    st = pd.DataFrame()
    t = yf.Ticker(x)
    st = t.history(start=y, end=z)
    st.index = pd.to_datetime(st.index).tz_localize(None)
    return(st)

def sdata(x, y, z):
    """
    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.

    Returns
    -------
    st : TYPE
        DESCRIPTION.

    """
    df = pd.DataFrame()
    t = yf.Ticker(x)
    df = t.history(start=y, end=z)['Close']
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

def vdata(x,y,z):
    """
    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.

    Returns
    -------
    st : TYPE
        DESCRIPTION.

    """
    df = pd.DataFrame()
    t = yf.Ticker(x)
    df = t.history(start=y, end=z)['Volume']
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

def odata(x,y,z):
    print("this function is under development")
    #"""
    #"""
    #latest optionsdata

    #tk = yf.Ticker(x)
    #options = tk.option_chain()
    #OptC = options.calls
    #OptC['Type'] = "Calls"
    #OptP = options.puts
    #OptP['Type'] = "Puts"
    #options = pd.concat([OptC,OptP])
    
    #options = options.reset_index(drop = True)
    
    #options['Expiration Date'] = "20"+opt['contractSymbol'][0].replace(symbol,"")[0:2] + "-" + opt['contractSymbol'][0].replace(symbol,"")[2:4] + "-" + opt['contractSymbol'][0].replace(symbol,"")[4:6]

    return(x+y+z)


class data_object:   
    
    def __init__(self, data = ""):
        
        """
        'E:/Python/weather.csv'
        'E:/Python/testcsv.cs'v''
        'E:/Python/tester.txt '
        'E:/Python/titanic.txt '
        'E:/Python/exceltest.xlsx '
        """
        
        self.today = date.today() 
        self.today = datetime(self.today.year,self.today.month,self.today.day) #today
       #self.week = datetime(self.today.year,self.today.month,self.today.day - 7) #today
        #self.month = datetime(self.today.year,self.today.month - 1,self.today.day) #today
        
        
        self.one = datetime(self.today.year-1,self.today.month,self.today.day) #one year ago
        self.three = datetime(self.today.year-3,self.today.month,self.today.day) #three years ago
        self.five = datetime(self.today.year-5,self.today.month,self.today.day) #five years ago
        self.ten = datetime(self.today.year-10,self.today.month,self.today.day) #ten years ago
        self.twenty = datetime(self.today.year-20,self.today.month,self.today.day) #twenty years ago
        
        
        if type(data) != str:
            
            self.data = data
            
            if type(self.data) == yf.Ticker:
            
                self.data = data.history(start='2020-01-01', end=self.today)
                self.data.index = pd.to_datetime(self.data.index).tz_localize(None)
  
        else:
            
            self.data = self.insert_data(data)
            
    def insert_data(self, data = "", interval = "1d", start_ = "2020-01-01", end_ = ""):
        
        if end_ == "":
            end_ = self.today
        
        
        if type(data) == str:
            
            if data != "":
            
                if "csv" in data[(len(data)-4):len(data)]:
                    f = open(data, "r") 
                    f = f.read()
                    f = f.split("\n")
        
                    b = 0
                    for i in range(len(f)):
                        if len(f[i]) == len(f[i+1]) and len(f[i]) == len(f[i+2]):
                            b = i
                            break
                    f = f[b:]
        
                    for i in range(len(f)):
                        if ";" in f[i]:
                            f[i] = f[i].split(";")
                        elif "," in f[i]:
                            f[i] = f[i].split(",")
                    
                    f = pd.DataFrame(f)
        
                elif "txt" in data[(len(data)-4):len(data)]:
                    f = open(data, "r")
                    f = f.read()
                    f = f.split("\n")
        
                    for i in range(len(f)):
                        if '"' in f[i]:
                            f[i] = f[i].replace('"','')
                        f[i] = f[i].split(",")
        
                    b = 0
                    for i in range(len(f)):
                        if len(f[i][0]) == len(f[i+1][0]) and len(f[i][0]) == len(f[i+2][0]):
                            #print(len(f[i]))
                            b = i
                            break
                    f = f[b:]
        
                    for i in range(len(f)):
                        if "   " in f[i][0]:
                            f[i] = f[i][0].split("   ")
                    
                    f = pd.DataFrame(f)
        
                elif "xlsx" in data[(len(data)-4):len(data)]:
                    f = pd.read_excel(data) 
                
                #elif "xlsm" in path[(len(path)-4):len(path)]:
                    
                else:
                    raise AttributeError("We don't know this file or filetype")
                    
                self.data = f
                    
                    
            else:
                self.data = pd.DataFrame()
            
        elif type(data) == list:
            self.data = data
            print("data has been read as a list type")
        
        elif type(data) == dict:
            self.data = data
            print("data has been read as a dictionary type")
        
        elif type(data) == np.array:
            self.data = data
            print("data has been read as a numpy array")
        
        elif type(data) == pd.DataFrame:
            self.data = data
            print("data has been read as a pandas DataFrame")
        
        elif type(data) == yf.Ticker:
            self.insert_price_data(data, interval, start_, end_)
            print("data has been read as yahoo finance Ticker")
                
    def server_SQL(self, TABLE_NAME = "", condition = ""):
        engine = sa.create_engine('mssql+pyodbc://localhost/TMRW?driver=SQL+Server+Native+Client+11.0')
        query = "SELECT * FROM [dbo].[" + TABLE_NAME + "]"
        if condition != "":
            query = query + " WHERE " + condition
        try:
            df = pd.read_sql(query, engine)
        except:
            print("This is not a table in the database")
        self.data = df
            
    def server_tables(self):
        engine = sa.create_engine('mssql+pyodbc://localhost/TMRW?driver=SQL+Server+Native+Client+11.0')
        query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' AND TABLE_CATALOG='TMRW'"
        df = pd.read_sql(query, engine)
        self.tables = df
        return(self.tables)
    
    def server_insert(self, TABLE_NAME, table):
        print("not done yet")
        
    def insert_price_data(self, data, interval = "1d", start_ = "2020-01-01", end_ = ""):
        
        if end_ == "":
            end_ = self.today
        
        if interval == "1d":
            self.data = data.history(start=start_, end=end_)
            self.data.index = pd.to_datetime(self.data.index).tz_localize(None)
        elif interval == "1h":
            self.data = yf.download(data.info['symbol'], start = self.month , end = self.today, interval = "1h")
            
        elif interval == "15m":
            self.data = yf.download(data.info['symbol'], start = self.week, end = self.today, interval = "15m")
        
        self.data['velocity'] = tf.returns(self.data['Close'])
        self.data['acceleration'] = tf.acceleration(self.data['Close'])
        self.data['volume velocity'] = tf.returns(self.data['Volume'])
        self.data['volume acceleration'] = tf.acceleration(self.data['Volume'])
        self.data['RSI'] = tf.RSI(self.data['Close'], 21)
        
        for i in [3, 5, 10, 21, 35, 60, 120]:
            txt = 'MA' + str(i)
            txt_i = 'MA' + str(i) + '_ind'
            txt2 = 'MSTD' + str(i)
            txt_mv = 'MV' + str(i)
            txt_std = 'MVSTD' + str(i)
            
            _ma = [0]
            _ma.extend(list(tf.ma(self.data['Close'], i).iloc[:, i-2]))
            
            ma_ind = []
            for j in range(len(_ma)):
                if _ma[j] < self.data.iloc[j,3]:
                    ma_ind.append(1)  # moving average is above
                elif _ma[j] >= self.data.iloc[j,3]:
                    ma_ind.append(-1) # moving average is below
                else:
                    ma_ind.append(0) # unknown
            
            std = [0]
            std.extend(list(tf.stdev(self.data['Close'], i).iloc[:, i-2]))
            
            self.data[txt] = list(_ma)
            self.data[txt_i] = ma_ind
            self.data[txt2] = list(std)
            self.data[txt_mv] = tf.moving_velocity_mean(self.data, i)
            self.data[txt_std] = tf.moving_velocity_std(self.data, i)
            
        for ind in [1, 3, 7, 21]:
            
            ind_list = []
            txt = "indicator" + str(ind)
            
            for i in range(len(self.data)):
                
                if i <= len(self.data) - 1 - ind:
                
                    if self.data.iloc[(i + ind),3] >= self.data.iloc[i,3]:
                        ind_list.append(1) # future value will be higher
        
                    elif self.data.iloc[(i + ind),3] < self.data.iloc[i,3]:
                        ind_list.append(0) # future value will be lower
                        
                else:
                    ind_list.append(-1) # error?
                    
            self.data[txt] = ind_list
        
        self.data = tf.UD_ind(self.data)
        self.data['Trend'] = list(tf.trend_ind(self.data)[0])
        self.data['Trend2'] = list(tf.trend_ind(self.data)[1])
    
        self.data['VWmean21'] = tf.volume_weighted_mean(self.data, window = 21)
        self.data['VWvelocity21'] = tf.volume_weighted_velocity(self.data, window = 21)
        
        for ind in [3, 5, 10, 21, 35, 60, 120]:
            
            ind_list = []
            txt = "indicator" + str(ind)
            
            for i in range(len(self.data)):
                
                if i <= len(self.data) - 1 - ind:
                
                    if self.data.iloc[(i + ind),3] >= self.data.iloc[i,3]:
                        ind_list.append(1) # future value will be higher
        
                    elif self.data.iloc[(i + ind),3] < self.data.iloc[i,3]:
                        ind_list.append(0) # future value will be lower
                        
                else:
                    ind_list.append(-1) # error?
                    
            self.data[txt] = ind_list
        
        self.data = tf.hidden_states(self.data)





        
















































   
    
