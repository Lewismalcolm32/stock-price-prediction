import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import time
import sqlite3

#--Part 4
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#--Part 5
import datetime


st.title("Real-time Stock Prices")
st.write("Welcome to the Real-Time Stock Price Predicition Application")

ticker = "AAPL"
data = yf.Ticker(ticker)

df = data.history(period="1y") # Example to get one month of data
# print(df.to_string())
# print(data.to_string())
# data = yf.download(ticker, period="1y")
data = yf.download(ticker, start="2024-01-01", end="2024-09-01")
# print(data.head(50))

#--COnnect to Database and create table
conn = sqlite3.connect('stock_data.db')
data.to_sql(ticker, conn, if_exists='replace')


#--Cleaning Data to create basic trend like Moving Average

data = pd.read_sql(f'SELECT * FROM {ticker}', conn, index_col='Date') #~~~ndex_col in .read_sql


#--Feature Engineering: Create a simple moving average (SMA)

data['SMA_10'] = data['Close'].rolling(window=10).mean()#~~~What's rolling; is it looking at the next ten prices and then gettign the mean using .mean()?
data['Return'] = data['Close'].pct_change()#~~~Is pct_change a function of a dataframe
#Adding new column "SMA_10" and "Return" to dataframe "data"

data.dropna(inplace=True)#~~~I wish  I could see what happens with we kept NA's in and why removing them is necessary


#--Defining column Target: 1 if price increase tomorrow, 0 otherwise
data['Target'] = (data['Close'].shift(-1) > data['Close'].shift(0)).astype(int)
# print(data.to_string())

#--Features and target variable:
#----Features being the data used as input necessary to train an ML model -- Data needs to be transfromed into features (Feature Engineering)
#-----What this Feature Engineering looks like is taking a column that can be used as a source of truth and adding a boolean column  next to, 
#this teaches the model what is expected for future predicitons


# pemdas --> mdas --> C: m then a; F: s then d
# 20 --> 70
# 20*2 == 40
# 40 + 30 == 70

# 70-30 == 40
# 40/2 == 20

features = ['SMA_10', 'Return']
X = data[features]
y = data['Target']
# print(y.to_string())

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, shuffle=False) #~~~IDK what she's doing here

model = LogisticRegression()        #~~~Look into the methods of the Logistic Regression Function (also what other models are offered out of I'm assuming scikit?)
model.fit(X_train, y_train)         #~~~Look into the methods of the Logistic Regression Function
predictions = model.predict(X_test) #~~~Look into the methods of the Logistic Regression Function

accuracy = accuracy_score(y_test, predictions) #~~IDK what accuracy_score is, or why it's important. 
print(f'Modell Accuracy: {accuracy:.2f}')


#--Real-Time Prediction Setup
def predict_next_day(stock_data, model):
	largest_data = stock_data[-1:]

	next_day_predicition = model.predict(latest_data[features])
	return next_day_predicition

#simulate real-time prediciton (replace with real-time pipeline)
today = datetime.datetime.now().strftime('%Y-%m-%d')
latest_data = yf.download(ticker, start=today, end=today, interval='1d')


#Append new data and make a predicition
data = data.apend(latest_data)
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['Return'] = data['Close'].pct_change()
data.dropna(inplace=True)

next_day_predicition = predict_next_day(data, model)

print(f'Next Day Predicition: {"Up" if next_day_predicition[0] == 1 else "Down"}')



#Testing












# ticker = "AAPL"  # Example ticker symbol
# print(data.info)
