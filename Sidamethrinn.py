import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load the data from the CSV file
data = pd.read_csv("/Program Files/PytonProject/Sidamethrin.csv")

# Convert the "Month" column to datetime format
data["Month"] = pd.to_datetime(data["Month"], format="%B %Y")

# Set the "Month" column as the index
data.set_index("Month", inplace=True)

# Visualize the data
plt.plot(data["Sales"])
plt.xlabel("Bulan")
plt.ylabel("Penjualan")
plt.title("Sidamethrin 400ml")
plt.show()

# Define the ARIMA parameters (p, d, q)
p = 8  # Autoregressive Order : Ketergantungan Dengan Masa Lalu
d = 0  # Degree of Differencing : Ketergantungan Dengan Siklikal / Seasonal
q = 8  # Moving Average Order : Fluktuasi Adaptasi Jangka Pendek

# Create and fit the ARIMA model
model = ARIMA(data["Sales"], order=(p, d, q))
model_fit = model.fit()

# Generate forecasts for the next 3 months
forecast = model_fit.predict(
    start=data.index[-1], end=data.index[-1] + pd.DateOffset(months=12), dynamic=True
)

# Plot the original data and forecasts
plt.plot(data["Sales"], label="Data Aktual")
plt.plot(forecast, label="Proyeksi")
plt.xlabel("Bulan")
plt.ylabel("Penjualan")
plt.title("Sidamethrin 400ml")
plt.legend()
plt.show()
