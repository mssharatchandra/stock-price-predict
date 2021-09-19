import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from keras.models import load_model
import streamlit as st
import pandas_datareader as data

start = '2015-01-01'
end = date.today().strftime("%Y-%m-%d")

st.title('📈🚀🌕Stock Price Prediction using Stacked LSTM Model💫') 

user_input = st.text_input("Enter the stock acronymn (refer yahoofinance for the acronymn eg:- AAPL for Apple.Inc)",'AAPL')

df = data.DataReader(user_input, 'yahoo', start , end)

#display this data to the user

st.subheader('💫Data from 2010 - Present💫')
st.write(df.describe())


#Visualisations

st.subheader('💫Closing price vs Time chart💫')
fig = plt.figure(figsize=(18,9))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('🔴100 Moving average of Closing price vs Time chart💫')

ma100=df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100,'r')
st.pyplot(fig)


st.subheader('🔴100 & 🟢200 Moving average of Closing price vs Time chart💫')

ma200=df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100,'r', label='100 Moving Avg')
plt.plot(ma200,'g', label='200 Moving Avg')
st.pyplot(fig)

st.subheader('💫Insight : Everytime the Red line crosses the Green line, there is a potential profit and vice versa🤑💸💰')

# Train Test Split

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

# we have to scale down the data for stack lstm input hence we import minmaxScalar from sklearn

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)


# Loading the model 
model = load_model('Latest_keras_model_10_epochs.h5')


# testing this model
past_100_days = data_training.tail(100)

final_df = past_100_days.append(data_testing, ignore_index = True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])


# converting these arrays into numpy arrays coz LSTM asks for numpy arrays only

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor

# RESULTS
st.subheader('💫Predicted price vs Market price💫')
fig2 = plt.figure(figsize = (12,6))
plt.plot(y_test, 'g', label = 'Market Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


st.text('This model has been trained on real time data scraped from Yahoo Financee')
 
st.text('and uses Stacked LSTM architecture')


st.subheader("💫💫💫Made with curiosity by [Sharat Chandra MS](https://www.linkedin.com/in/sharat-chandra-ms-a17457197/)🏄🏼🏄🏼🏄🏼")







# final graph x axis time change 

#explore about forecasting 

