"""
--- The Stock Price Predictor ---
created by: León La Marca

This programs aim is to predict the stock market price of a company using a simple neural network with 3 layers
(in this program a sequential model is used). The neural network trains with the data provided by a finance api
(e.g. Yahoo) and will then predict what the price of a stock will be on the following day.
It is important that this is not any financial advise, I do not recommend trading based on the information
you get from that program. It is for educational purpose only!
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import pandas_datareader as pdr

from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM

""""
First the Programm should load the actual pricedata of a company using the yahoo finance page.
"""
#Load The Data:
company = '\^AAPL' #\^ because we have to escape the \^

#start = dt.datetime(2012,1,1)
#end = dt.datetime(2022,1,1)

data = pdr.DataReader(company, data_source='yahoo', start='2012-01-01', end='2020-01-01') #yahoo = finance API. You can use any finance API of your choice!
#There is a Traceback error with the DataReader which i couldn´t fix :/

"""
In order to work with the Data we loaded before, we have to modify and prepare the Data a little bit.
This Program predicts the Stockprice for the next day, using the information of the past 60 days.
"""
#Prepare the Data:
scaler = MinMaxScaler(feature_range=(0,1)) #scales the values (min Value and Max Value) between 0-1
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1)) #Transform the Data (Closing Price!)

prediciton_days = 60 #sets the amout days on which the prediction is based

x_train = []
y_train = []

for x in range(prediciton_days, len(scaled_data)): #from the 60th till the last day
    x_train.append(scaled_data[x - prediciton_days:x]) #60 days
    y_train.append(scaled_data[x, 0])

x_train = np.array(x_train) #convert in np arrays
y_train = np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]), 1) #adds one dimension --> so that the neural network (NN) is able to work with the data

"""
Now we have to build the a Neural Network, which learns how to handle the data and then predicts the stock price
for the next day. In this programm i use a Sequential model, which fits perfect for this example.
"""
#Build the Model:
model = Sequential() #that model is commonly used if you have single values as inputs and outputs in a NN.

model.add(LSTM(units=50, return_sequences=True, input_shape = (x_train.shape[1], 1))) #add layer (LSTM = Long short-term Memory)
model.add(Dropout(0.2))
model.add(LSTM(units=50), return_sequences=True) #add layer
model.add(Dropout(0.2))
model.add(LSTM(units=50)) #add layer
model.add(Dropout(0.2))
model.add(Dense(units=1)) #outputs the prediction of the next closing value

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32) #epochs=25 --> model sees the data 25 times; batch_size = number of training models used in one iteration

'''Test the Model accuracy on existing Data'''

"""
In the next section, we will load the actual data of our company till today, so we can predict the stock price
of tomorrow. 
"""
#Load Test Data:
test_start = dt.datetime(2022, 1, 1)
test_end = dt.datetime.now()

test_data = web.DataReader(company, 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values #Close --> closing Values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)  #concatenates the actual closing values with the test closing values

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediciton_days:].values #determines the data that is used to predict the next closing value.
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs) #scale it down

"""
First we will try to make some predictions based on the Test Data.
"""
#Make Predictions on Test Data:

x_test = []

for x in range(prediciton_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediciton_days:x], 0)

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]), 1) #adds one dim

predicted_prices = model.predict(x_test) # predicted prices --> scales --> retransform!
predicted_prices = scaler.inverse_transform(predicted_prices)


"""
In order to have a better and clearer image of how our Program is working, we want to plot a graph,
where we see the actual stock price and the stock price that is predicted by our model.
"""
#Plot the Test Predictions:
plt.plot(actual_prices, color=black, label=f"Actual {company} Price")
plt.plot(predicted_prices, color=green, label=f"Predicted {company} Price")
plt.title(f"{company} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{company} Share Price")
plt.legend
plt.show()


"""
The Last step is to predict what the stock price will be tomorrow, and then print it into the console.
"""
#Predict next day:

real_data = [model_inputs[len(model_inputs) + 1 - prediciton_days:len(model_inputs + 1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1)) #add one dim

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Preditcion: {prediction}") #print the prediction into the console
