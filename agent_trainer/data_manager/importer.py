import os
import pandas as pd
import numpy as np


def get_df(path):
  df = pd.read_json(path, orient= 'values' )
  #df = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
  df.columns = ["Date", "Open", "High", "Low", "close", "Volume"]
  

  df['Date'] = pd.to_datetime(df['Date'],unit='ms')
  df.set_index('Date', inplace=True)
  
  print(df.head())
  
  # # check if directory is empty
  # if not os.path.isdir() or len(os.listdir(path)) == 0:
  #   print("ERROR Directory does not exist or is empty")
  #   return
    
  # for filename in os.listdir(path):
  #     f = os.path.join(path, filename)
      
  #     # checking if it is a file
  #     if os.path.isfile(f):
  #         print(f)
  return df

# get_df("C:\\Users\\vandenbb\\source\\vscode\\deepfreq\\user_data\\data\\binance\\BTC_USDT-1m.json")
