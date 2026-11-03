# ARIMA versus Prophet on daily COVID-19 cases in Nigeria, 1 Jan to 31 Dec 2021
# (365 observations, NCDC data)
#
# Workflow follows the paper: N. G. Okoacha and P. Osatohanmwen (2024), "Analysis of
# Nigeria's COVID-19 Data using Time Series and Machine Learning Models: Imperatives
# for Actions into Possible Future Pandemic", IEEE SEB4SDG 2024.
#
# Data: put the NCDC daily cases in "ncdc_covid_2021.csv" with two columns:
#   date  (YYYY-MM-DD)   and   cases  (daily confirmed cases)
# Reconstructed from the paper's description, not the original script.
# Check the settings marked CHECK against your original analysis.

library(forecast)   # Arima, auto.arima, BoxCox, accuracy
library(prophet)    # Facebook Prophet
library(tseries)    # adf.test

#-------------------------------------------------------------
# 1. Data
#-------------------------------------------------------------
covid <- read.csv("ncdc_covid_2021.csv")
covid$date <- as.Date(covid$date)
covid <- covid[order(covid$date), ]
stopifnot(nrow(covid) == 365)               # 1 Jan to 31 Dec 2021

cases <- ts(covid$cases, frequency = 1)
summary(covid$cases)
plot(covid$date, covid$cases, type = "l",
     xlab = "Date", ylab = "Daily COVID-19 cases, Nigeria (2021)")

#-------------------------------------------------------------
# 2. ARIMA
#-------------------------------------------------------------
# Stationarity: ADF test, then first differencing
adf.test(cases)
d_cases <- diff(cases, differences = 1)
adf.test(d_cases)
plot(d_cases, ylab = "Differenced daily cases")

# Search over (p, 1, q) and keep the model with the smallest AIC
# CHECK: the paper reports ARIMA(3,1,4) as best by AIC. The search range is an assumption.
best <- NULL
for (p in 0:5) {
  for (q in 0:5) {
    fit <- try(Arima(cases, order = c(p, 1, q), method = "ML"), silent = TRUE)
    if (inherits(fit, "try-error")) next
    if (is.null(best) || fit$aic < best$aic) best <- fit
  }
}
print(best)

# Residual diagnostics (paper reports a Ljung-Box p-value of 0.0946)
# CHECK: the lag is an assumption
Box.test(residuals(best), lag = 10, type = "Ljung-Box")
checkresiduals(best)

# In-sample accuracy on the original scale
arima_acc <- accuracy(best)
arima_mae  <- arima_acc[1, "MAE"]
arima_rmse <- arima_acc[1, "RMSE"]

# 90-day forecast
arima_fc <- forecast(best, h = 90)
plot(arima_fc, main = "ARIMA: 90-day forecast of daily cases")

#-------------------------------------------------------------
# 3. Prophet
#-------------------------------------------------------------
# De-noise with a Box-Cox transformation, as in the paper.
# CHECK: lambda is estimated here; use the value from your original analysis.
# Box-Cox needs positive values, so add 1 first if any day has zero cases.
shift  <- if (any(covid$cases <= 0)) 1 else 0
lambda <- BoxCox.lambda(covid$cases + shift)
df <- data.frame(ds = covid$date,
                 y  = BoxCox(covid$cases + shift, lambda))

m <- prophet(df,
             daily.seasonality  = TRUE,
             weekly.seasonality = TRUE,
             yearly.seasonality = TRUE)

# Fitted values and 90-day forecast, back-transformed to case counts
future   <- make_future_dataframe(m, periods = 90)
pred     <- predict(m, future)
pred$cases_hat <- InvBoxCox(pred$yhat, lambda) - shift

fitted_part <- pred[seq_len(nrow(covid)), ]
prophet_mae  <- mean(abs(covid$cases - fitted_part$cases_hat))
prophet_rmse <- sqrt(mean((covid$cases - fitted_part$cases_hat)^2))

plot(m, pred)

#-------------------------------------------------------------
# 4. Comparison
#-------------------------------------------------------------
results <- data.frame(Model = c("ARIMA", "Prophet"),
                      MAE   = c(arima_mae,  prophet_mae),
                      RMSE  = c(arima_rmse, prophet_rmse))
print(results)
# The paper reports: ARIMA(3,1,4) MAE 123.3892, RMSE 240.9052;
#                    Prophet       MAE 129,      RMSE 266.1483.
# Your numbers may differ slightly if the data or settings differ from the original.
