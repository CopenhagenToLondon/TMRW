import yfinance as yf
import pandas as pd
import sqlalchemy as sa
from binance.enums import *
from datetime import datetime, date, timedelta, timezone

today = date.today() 
today = datetime(today.year,today.month,today.day) + timedelta(days = +1) 
day7 = datetime(today.year,today.month,today.day) + timedelta(days = -25) 
todays = today.strftime("%d-%m-%Y")
day7 = day7.strftime("%d-%m-%Y")

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


symbol = "MSFT"
t = yf.Ticker(symbol)
t.info

t.history(start=y, end=z)

options = t.option_chain()
OptC = options.calls
OptC['Type'] = "Calls"
OptP = options.puts
OptP['Type'] = "Puts"
options = pd.concat([OptC,OptP])

options = options.reset_index(drop = True)

options['Expiration Date'] = "20"+options['contractSymbol'][0].replace(symbol,"")[0:2] + "-" + options['contractSymbol'][0].replace(symbol,"")[2:4] + "-" + options['contractSymbol'][0].replace(symbol,"")[4:6]
options

import requests

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
url = 'https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&symbol=IBM&apikey=demo'
r = requests.get(url)
data = r.json()

print(data)


api_key = 'vBrHMZUCDxlfNiAreBj02aUrymSFMGyl1AwJNAP0O7qVlvh07Drq7qQwfQlbYGeS'
api_secret = 'SdzCkiFE5zNjkSvptRVpxQBxlUWZWYrjnRGLp5tzhzJiat79vymOH127zaKHnnCh'

from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
client = Client(api_key, api_secret)

tick = "BTCUSDT"

klines = client.get_historical_klines(tick, Client.KLINE_INTERVAL_1HOUR, day7, todays)
data = pd.DataFrame(klines, columns = ['Open time', 'Open','High','Low','Close','Volume','Close time','Quote asset volume','Number of trades','Taker buy base asset volume','Taker buy quote asset volume', 'Ignore'])

data.Open = data.Open.astype(float)
data.High = data.High.astype(float)
data.Low = data.Low.astype(float)
data.Close = data.Close.astype(float)
data.Volume = data.Volume.astype(float)
data['Number of trades'] = data['Number of trades'].astype(float)
data['Taker buy base asset volume'] = data['Taker buy base asset volume'].astype(float)
data['Taker buy quote asset volume']= data['Taker buy quote asset volume'].astype(float)
data = data[['Open','High','Low','Close','Volume', 'Number of trades','Taker buy base asset volume','Taker buy quote asset volume']]
