""""
--- The Stock Price Predictor ---
created by: León La Marca
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM

""""
First the Programm should load the actual pricedata of a company using the yahoo finance page.
"""
#Load The Data:
company = 'FB'

start = dt.datetime(2012,1,1)
end = dt.datetime(2022,1,1)

data = web.DataReader(company, 'yahoo', start, end) #yahoo = finance API, man kann auch andere verwenden!

"""
In order to work with the Data we loaded before, we have to modify and prepare the Data a little bit.
This Program predicts the Stockprice for the next day, using the information of the past 60 days.
"""
#Prepare the Data:
scaler = MinMaxScaler(feature_range=(0,1)) #skaliert alle Werte (min Value und Max Value) zwischen 0-1
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1)) #Transform the Data (Closing Price!)

prediciton_days = 60 #Auf wievielen Tagen soll die Prediction passieren

x_train = []
y_train = []

for x in range(prediciton_days, len(scaled_data)): #von 60 bis zum letzten Tag
    x_train.append(scaled_data[x - prediciton_days:x]) #60 Tage
    y_train.append(scaled_data[x, 0])

x_train = np.array(x_train) #In np Arrays convertieren
y_train = np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]), 1) #adds one dimension --> damit man dann damit im Neural Network (NN) arbeiten kann

"""
Now we have to build the a Neural Network, which learns how to handle the data and then predicts the stock price
for the next day. In this programm i use a Sequential model, which fits perfect for this example.
"""
#Build the Model:
model = Sequential() #model welches geeignete ist wenn man einzelne Values als input und output bei einem NN hat.

model.add(LSTM(units=50, return_sequences=True, input_shape = (x_train.shape[1], 1))) #Layers adden (LSTM = Long short-term Memory)
model.add(Dropout(0.2))
model.add(LSTM(units=50), return_sequences=True) #Layers adden
model.add(Dropout(0.2))
model.add(LSTM(units=50)) #Layers adden
model.add(Dropout(0.2))
model.add(Dense(units=1)) #Prediction of the next Closing Value

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32) #epochs=25 --> model sieht die Daten 25 mal; batch_size = number of training models used in one iteration

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

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)  #verkettet die actual closing values mit den test closing values

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediciton_days:].values #bestimmt womit das Model arbeitet um Predictions zu machen
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

predicted_prices = model.predict(x_test) # predicted prices --> scales --> rücktransformieren!
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
print(f"Preditcion: {prediction}")