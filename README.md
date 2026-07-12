# Wizz Air Passenger Forecast

<div align="center">

<img src="wizzair_logo.png" width="250">

<div align="left">
  
## Project Overview

This project focuses on forecasting the monthly number of passengers of **Wizz Air** using time series modelling techniques.

The analysis is based on monthly passenger data covering the period from **April 2014 to June 2026**. The main objective was to build a forecasting model capable of predicting future passenger volumes and identifying the expected growth trend of the airline.

## Methodology

The forecasting model is based on a **linear regression model with ARIMA errors**. This approach combines:
- a regression component to capture the influence of explanatory variables,
- an ARIMA error structure to model autocorrelation and time-dependent patterns in the residuals.

The model was estimated using historical monthly passenger data and then used to generate forecasts for the following 12 months.

## Forecast Results

According to the model predictions:

- In **July 2026**, the number of Wizz Air passengers is expected to **exceed 8 million passengers**.
- In **July 2027**, the forecast indicates that the number of passengers will **approach 9 million passengers**.

These results suggest a continued growth trend in Wizz Air passenger traffic.

## Tools and Technologies

The project was developed in **R** using packages for:
- time series analysis,
- regression modelling,
- ARIMA forecasting,
- data visualization.

## Repository Contents

The repository includes:
- source code for data preparation and modelling,
- exploratory data analysis,
- model estimation,
- forecast generation,
- visualization of historical data and future predictions.

## Author

Project developed as a time series forecasting analysis of Wizz Air passenger traffic.

The main results of the analysis, including forecast values and visualizations, are available in the file **`WizzAir_forecast.pdf`**.
