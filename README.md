# Stock Price Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset

The task involves forecasting Google stock prices over time using RNN model for future stock price prediction. The dataset is a time series dataset consisting values of dates, opening stock rates, closing stock rates, High and low stock values. Based on the attributes given for the dataset, a stock price prediction model is to be developed.
![image](https://github.com/SarankumarJ/rnn-stock-price-prediction/assets/94778101/3f416bfb-5f23-4f64-8f96-58d6ea62021f)


## Design Steps

### Step 1:
Prepare training data and preprocess it using MinMaxScaler.

### Step 2:
Define and compile the RNN model architecture.

### Step 3:
Train the RNN model using the prepared training data.

### Step 4:
Prepare test data, preprocess it, and predict using the trained model.

### Step 5:
Evaluate model performance and visualize the results.



## Program
#### Name: Sarankumar J
#### Register Number: 212221230087

```py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential
from sklearn.metrics import mean_squared_error

# Load training dataset
dataset_train = pd.read_csv('trainset.csv')
train_set = dataset_train.iloc[:, 1:2].values

# Scale the training set
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(train_set)

# Prepare training data with a window of 60 time steps
X_train_array = []
y_train_array = []
for i in range(60, 1259):
    X_train_array.append(training_set_scaled[i-60:i, 0])
    y_train_array.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# Define and compile the model
length = 60
n_features = 1
model = Sequential()
model.add(layers.SimpleRNN(50, input_shape=(length, n_features)))
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mse')

print("Sarankumar J")
print("212221230087")

model.summary()

# Train the model
model.fit(X_train1, y_train, epochs=100, batch_size=32)

# Test Data
dataset_test = pd.read_csv('testset.csv')
test_set = dataset_test.iloc[:, 1:2].values

# Combine training and test set for preparing inputs
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total.values
inputs = inputs.reshape(-1, 1)
inputs_scaled = sc.transform(inputs)

X_test = []
y_test = []
for i in range(60, 1384):
    X_test.append(inputs_scaled[i-60:i, 0])
    y_test.append(inputs_scaled[i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Make predictions on the test data
predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)

print("Sarankumar J")
print("212221230087")
plt.plot(np.arange(0, 1324), inputs[:1324], color='red', label='Test (Real) Google stock price')
plt.plot(np.arange(60, 1384), predicted_stock_price, color='blue', label='Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
```

## Output

### True Stock Price, Predicted Stock Price vs time

![image](https://github.com/SarankumarJ/rnn-stock-price-prediction/assets/94778101/86243c7c-8356-49f8-856f-b085d2d5ffe1)


### Mean Square Error

![image](https://github.com/SarankumarJ/rnn-stock-price-prediction/assets/94778101/a32d3e80-887c-48c0-b690-5eb41c76a315)


## Result
Thus the stock price is predicted using Recurrent Neural Networks successfully.
