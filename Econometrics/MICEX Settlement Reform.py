##### Master 203 - Python Programming - October 2017
##### Sebastien Jérémie DAVID

##### Assignment: Moscow Stock Exchange Settlement Reform
##### Python Codes for Data Processing and Panel Regression

# Works on Python 3.7 but the pandas libray must be an older
# version (pre-July 2017) since the code here utilizes the OLS
# and panelOLS functions

#%%
##### Packages
# Handling files (cf. task A)
import os
# Zip extraction (cf. task B and C.1)
import gzip, shutil
# Dataframe manipulation and panel regression (cf. tasks B, C.1, E)
import pandas as pd
# Numerical functions (cf. task C.5)
import numpy as np
import math
### Package for datetime objects (cf. task C.9)
import datetime
# Package for making copies of objects
import copy
# Package for making graphical plots (cf. task D)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.style.use('default')

#%%
##### Task A

# Folder path of Moscow Stock Exchange (MSE) intraday data
path = "C:/Users/Sebastien David/Documents/Master 2/Semester 1/Python/DM/Intraday"
directory = os.listdir(path)

# Load all MSE intraday files into a list `list_files`
list_files = []
for file in directory:
    list_files.append(file)

#%%
##### Task B

# Unzips (specifically) the first of 4529 gz files
# Creates a temporary CSV file that can be imported
# into pandas for data manipulation. The CSV copy is
# stored in the MSE Intrady folder.
filename = path+"/"+list_files[0] # first file
with gzip.open(filename, 'rb') as f_in, open(filename+".csv", 'wb') as f_out:
    shutil.copyfileobj(f_in , f_out)

# Loads the csv into a Pandas DataFrame
df = pd.read_csv(filename+".csv")

# Deletes the csv copy from the MSE Intraday folder to
# free up space and avoid consumming too much memory
os.remove(filename+".csv")

#%%
##### Task C

# QUESTION 1
# The objective is to aggregate daily data from the
# Moscow Stock Exchange (MSE) stocks from a list of
# 4529 zip files. Each zip file contains hourly stock
# market data for 1 MSE stock and 1 month.

# 10 procedures will be executed to compute new metrics
# and clean up each pandas-loaded CSV file, which will
# lead to an aggregated daily MSE dataset with total
# traded volume, total number of quotes/trades, median
# quoted spread and mean volatility for each stock per
# day, i.e. multi-indexed pandas dataFrame `DataDay`.

# Instead of loading all the Intraday files at once, we
# run only the 10 procedures for 1 zip file at a time
# (meaning 1 month for 1 stock) since we are only really
# interested in the daily data. Thus by aggregating data
# from a high-frequency to a daily time series, we reduce
# the total number of rows per pandas-loaded zip file from
# roughly 10 000 to 22 (trading days).


# Initialize aggregated daily time series dataset
DataDay = pd.DataFrame([])

# Number of files
nb_files = len(list_files)

# For loop procedure
for m in range(nb_files):
    
    # Unzip gz file into CSV
    filename = path+"/"+list_files[m]
    
    with gzip.open(filename, 'rb') as f_in, open(filename+".csv",
                  'wb') as f_out:
        shutil.copyfileobj(f_in , f_out)
    
    # Load MSE data for 1 stock and 1 day into a pandas DataFrame
    # temp is for temporary since we are going to load a new temp
    # file everytime we change zip file
    temp = pd.read_csv(filename+".csv")
    
    # Delete csv file for memory space
    os.remove(filename+".csv")
    
    # QUESTION 2
    # We define the portion of the dataframe corresponding
    # to the stock tickers and order books (here #RIC) as
    # a dataframe "q".
    
    # The column #RIC contains market indicators and stock
    # tickers "ZVEZ.MM". For the market indicators, we are
    # only interested in the "MM" portion, thus we use the
    # Python apply(lambda x:) function in order to automa-
    # tically loop across all the rows of #RIC and split
    # the #RIC ticker into 2 components: the Market compo-
    # nent (MM) and the Symbol component (SVEZ).
    
    # We split the ticker by cutting the #RIC ticker string
    # at "."
    q = temp.loc[:,'#RIC'].apply(lambda x: x.split('.'))
    
    # Spliting a string creates a list composed of the split
    # components. Since we have 2 components in the split li-
    # st and the market indicator is the 2nd element, we use
    # apply on q with x[1]. We thus create a new column of
    # only market indicators.
    temp['Market'] = q.apply(lambda x: x[1])
    
    # QUESTION 3
    # Same as before, except we extract the 1st component of
    # the split list which contains the stock indicator
    temp['Symbol'] = q.apply(lambda x: x[0])
    
    # QUESTION 4
    temp['Traded Volume'] = temp['VWAP'] * temp['Volume']
    
    # QUESTION 5
    # Function we use to add the "nan" value when the Close
    # Bid and Ask prices contain a 0.0 value, and change no-
    # thing if the prices are not equal to 0.0.
    def add_nan(value):
        if value == 0.0:
            res = np.nan
        else:
            res = value
        return res

    # Use the add_nan() function to replace all 0 values with
    # "NaN" values in the existing Close Bid and Ask columns
    temp['Close Bid'] = temp['Close Bid'].apply(lambda x: add_nan(x))
    temp['Close Ask'] = temp['Close Ask'].apply(lambda x: add_nan(x))

    # QUESTION 6
    # Create a new column for the quoted spread
    temp['QSpread'] = 100 * (temp['Close Ask'] - temp['Close Bid'])/(0.5*
                    (temp['Close Ask'] + temp['Close Bid']))
    
    # QUESTION 7
    # Code for computing the number of quotes
    temp['No. Quotes'] = temp['No. Bids'] + temp['No. Asks']
    
    # QUESTION 8
    # Code for computing the range-based volatility
    k = 100/(2.0 * math.sqrt(np.log(2.0)))
    temp['Volatility'] = k * np.log(temp['High']/temp['Low'])
    
    # QUESTION 9
    # We want to convert time and date indices from
    # string values to datetime objects in Python, which
    # will allow for easier indexing and allow us to
    # implement algorithms that require the stock to be
    # "before" a given date (e.g. 30th August 2013). We
    # are going to use the striptime function to convert
    # string values into datetime objects.
    
    # For dates, we want the format '%d-%b-%Y': 30/08/2013
    date = temp['Date[G]']
    temp['Date'] = date.apply(lambda x: 
        datetime.datetime.strptime(x, '%d-%b-%Y').date())
    
    # For dates, we want the format '%H:%M:%S.%f': 04:30:00.000
    time = temp['Time[G]']
    temp['Time'] = time.apply(lambda x:
        datetime.datetime.strptime(x, '%H:%M:%S.%f').time())
    
    # QUESTION 10
    # Code that creates a dummy column T+0. For each row i,
    # if the market indicator is equal to 'MM' and the date
    # is before 30th August 2013, put 1.0, else 0.0.
    
    # There is no need to define an initial column temp['T+0']
    # = np.nan. Actually writing temp.loc[i,'T+0'] when ['T+0']
    # never existed before creates it when you run the following
    # loop:
    for i in range(len(temp)):
        if temp.loc[i,'Market'] == 'MM' and temp.loc[i,'Date'] < datetime.date(2013, 8, 30):
            temp.loc[i,'T+0'] = 1.0
        else:
            temp.loc[i,'T+0'] = 0.0
    
    # QUESTION 11
    # We use the groupby() function on temp in order to aggregate
    # hourly data of volume, number of quotes/trades, volatility &
    # quote spreads per day / stock / status during the settlement
    # reform (either 1.0 or 0.0) / market ticker
    multi_index = ['Date','Symbol','Market','T+0']
    final = temp.groupby(multi_index).agg({'Traded Volume':'sum',
                                           'No. Quotes':'sum',
                                           'QSpread':'mean',
                                           'Volatility':'mean',
                                           'No. Trades':'sum'})
    
    DataDay = DataDay.append(final)

#%%
##### Task D

# Copy of DataDay that is reindexed for ease in data manipulation
# with adjustments such as dropping the remaining NaN values
data = copy.copy(DataDay)
data = data.reset_index()

# Format X-axis to match the assignment example plot unit scales
data['Traded Volume'] = data['Traded Volume'] * 2.0 / 1000000000

# We select the 13 MSE stocks of interest
W1=['FEES','SBER','GAZP','MOEX','LKOH','GMKN','NVTK',
    'ROSN','HYDR','CHMF','SNGS','URKA','VTBR']

data = data[data['Symbol'].isin(W1)]

# We drop all nan values to avoid counting weekends/holidays
data = data.dropna()

# Time series of stocks before the MSE settlement reform (T+0)
data1 = data[data['T+0']==1.0]
data1 = data1.groupby(['Date']).agg({'QSpread':'median',
                                     'Traded Volume':'sum',
                                     'No. Quotes':'sum',
                                     'No. Trades':'sum'})

# Time series of stocks after the MSE settlement reform (T+2)
data2 = data[data['T+0']==0.0]
data2 = data2.groupby(['Date']).agg({'QSpread':'median',
                                     'Traded Volume':'sum',
                                     'No. Quotes':'sum',
                                     'No. Trades':'sum'})

# To ensure dates are labelled in format "Month Year"
date_format = mdates.DateFormatter('%b %Y')
locator = mdates.MonthLocator()

# The following plots are those generated in the PDF
# assignment report

# Plot 1: Total traded volume for each day
# 1. Basics
plt.figure(figsize=(16,6))
plt.plot(data1[['Traded Volume']],color='blue',label="T+0 Stocks",linewidth=2)
plt.plot(data2[['Traded Volume']], color='green',label="T+2 Stocks",linewidth=2)
plt.legend(fontsize=12)
plt.title('Figure 1 - Total daily traded volume for 13 MICEX stocks', fontsize=11)
# 2. X and Y-axis
plt.ylabel('Volume (rubles M)', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.xticks(rotation=15)
plt.grid(True)
plt.gca().xaxis.set_major_formatter(date_format) # To ensure dates
plt.gca().xaxis.set_major_locator(locator)       # are labelled in
plt.gcf().autofmt_xdate()                        # months
# 3. MSE settlement reform dates (March 25th - August 30th)
plt.axvline(x=datetime.datetime(2013, 3, 24), linewidth=2.5, color="black")
plt.axvline(x=datetime.datetime(2013, 8, 29), linewidth=2.5, color="black")
# Results
plt.plot()

# Plot 2: Median quoted spread for each day
plt.figure(figsize=(8,6))
plt.plot(data1[['QSpread']],color='blue',label="T+0 Stocks",linewidth=2)
plt.plot(data2[['QSpread']], color='green',label="T+2 Stocks",linewidth=2)
plt.legend(fontsize=12)
plt.title('Figure 2 - Daily median quoted spread for 13 MICEX stocks', fontsize=11)
plt.ylabel('Median Quoted Bid-Ask Spread', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.xticks(rotation=15)
plt.grid(True)
plt.gca().xaxis.set_major_formatter(date_format)
plt.gca().xaxis.set_major_locator(locator)
plt.gcf().autofmt_xdate()
plt.axvline(x=datetime.datetime(2013, 3, 24), linewidth=2.5, color="black")
plt.axvline(x=datetime.datetime(2013, 8, 29), linewidth=2.5, color="black")
plt.plot()

# Plot 3: Total number of quotes and trades for each day
plt.figure(figsize=(8,6))
# 1. Plots
plt.plot(data1[['No. Quotes']],color='blue',
         label="T+0 (Quotes)",linewidth=2)
plt.plot(data2[['No. Quotes']], color='green',
         label="T+2 (Quotes)",linewidth=2)
plt.plot(data1[['No. Trades']],color='indigo',
         label="T+0 (Trades)",linestyle='--')
plt.plot(data2[['No. Trades']], color='yellowgreen',
         label="T+2 (Trades)",linestyle='--')
# 2. Legend
plt.legend(fontsize=10)
plt.title('Figure 3 - Daily total number of quotes and trades for 13 MICEX stocks', fontsize=11)
plt.xlabel('Time', fontsize=12)
plt.xticks(rotation=15)
plt.grid(True)
plt.gca().xaxis.set_major_formatter(date_format)
plt.gca().xaxis.set_major_locator(locator)
plt.gcf().autofmt_xdate()
plt.axvline(x=datetime.datetime(2013, 3, 24), linewidth=2.5, color="black")
plt.axvline(x=datetime.datetime(2013, 8, 29), linewidth=2.5, color="black")
# 3. Results
plt.plot()

#%%
##### Task E

# Add a time series dummy variable column ('T+2') where the dummy
# takes 1.0 for stocks traded after the end of the MSE settlement
# reform on 30th August 2013, and 0.0 before the settlement reform
# of 25th March 2013
data['T+2'] = np.abs(-1+data['T+0'])

# From the set of 25 liquid stocks, we create 2 new time series
# datasets: one from January 1 to March 24 (pre-settlement re-
# form) and one from September 1 to December 31 (post settle-
# ment reform)
panel_1 = copy.copy(data[data['Date']<datetime.date(2013, 3, 25)])
panel_2 = copy.copy(data[data['Date']>datetime.date(2013, 8, 30)])
panel = panel_1.append(panel_2, ignore_index=True)

# The data is organized as a panel dataset where for each day (t)
# we have of the 13 stocks (i), for which we report their mean quo-
# ted spread of the day and the 'T+2' dummy variable
panel = panel.groupby(['Date','Symbol']).agg({"QSpread":"median","T+2":"min"})

# Fixed Effects Panel Regression Model
pmodel = pd.stats.plm.PanelOLS(y=panel['QSpread'], x=panel[['T+2']],
                               entity_effects=True, time_effects=False)

# Print results of fixed effects model
print(pmodel.summary)

#%%
# Bonus: comparaison with OLS Regression
panel_ols = panel.reset_index()

model = pd.stats.plm.OLS(y=panel_ols['QSpread'], x=panel_ols[['T+2']])
print(model.summary)