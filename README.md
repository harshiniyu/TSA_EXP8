# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date: 14/10/2025


### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

# --- Load Data ---
data = pd.read_csv("silver.csv")

# Check column names
print("Columns:", data.columns)

# Select 'Date' and 'USD' columns and rename 'USD' to 'Price'
data = data[['Date', 'USD']].rename(columns={'USD': 'Price'})

# Convert Date column to datetime
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

print("Shape of the dataset:", data.shape)
print("First 10 rows:")
print(data.head(10))

# --- Original Data Plot ---
plt.figure(figsize=(12, 6))
plt.plot(data['Price'], label='Original Silver Price')
plt.title('Original Silver Price Data')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()

# --- Moving Averages ---
rolling_mean_5 = data['Price'].rolling(window=5).mean()
rolling_mean_10 = data['Price'].rolling(window=10).mean()

plt.figure(figsize=(12, 6))
plt.plot(data['Price'], label='Original Data', color='blue')
plt.plot(rolling_mean_5, label='5-Day Moving Average')
plt.plot(rolling_mean_10, label='10-Day Moving Average')
plt.title('Moving Averages of Silver Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()

# --- Resample Monthly ---
data_monthly = data.resample('MS').mean()  # Monthly start

# --- Scaling ---
scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(),
    index=data_monthly.index
)

# Add 1 to handle multiplicative seasonality
scaled_data = scaled_data + 1

# --- Train-Test Split (80/20) ---
x = int(len(scaled_data) * 0.8)
train_data = scaled_data[:x]
test_data = scaled_data[x:]

# --- Holt-Winters Model ---
model = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()

test_predictions = model.forecast(steps=len(test_data))

# --- Plot Train vs Test ---
plt.figure(figsize=(12, 6))
train_data.plot(label='Train Data')
test_data.plot(label='Actual Test Data')
test_predictions.plot(label='Predicted')
plt.title('Holt-Winters Forecast (Train vs Test)')
plt.legend()
plt.grid()
plt.show()

# --- RMSE ---
rmse = np.sqrt(mean_squared_error(test_data, test_predictions))
print("Root Mean Squared Error (RMSE):", rmse)

# --- Forecast for Future (Next 12 months) ---
future_model = ExponentialSmoothing(
    scaled_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()

future_forecast = future_model.forecast(steps=12)

plt.figure(figsize=(12, 6))
scaled_data.plot(label='Historical Data')
future_forecast.plot(label='Next 12 Months Forecast')
plt.title('Future Forecast of Silver Price')
plt.xlabel('Date')
plt.ylabel('Scaled Price')
plt.legend()
plt.grid()
plt.show()
```
### OUTPUT:
<img width="774" height="351" alt="image" src="https://github.com/user-attachments/assets/1c1c769c-e4d5-44fd-972f-96f5b1a37997" />
<img width="996" height="547" alt="download" src="https://github.com/user-attachments/assets/ff622eae-ccc7-448d-b4ce-ce1631ad2eb2" />

<img width="981" height="547" alt="download" src="https://github.com/user-attachments/assets/876a7b34-48cd-4ddd-88c5-30cb15472ff3" />

<img width="968" height="520" alt="image" src="https://github.com/user-attachments/assets/9774f008-d17d-4253-bc2d-e57a509ff1f9" />

### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
