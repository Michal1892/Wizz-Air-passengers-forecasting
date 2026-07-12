---
  title: "Linear regression model with ARIMA errors for forecasting the number of Wizz Air passengers"
author: "Michał Porczyński"
date: "2026-07-09"
always_allow_html: true
output: pdf_document
header-includes:
  - \usepackage{float}
---
  
  The main goal of this paper is to build a model for forecasting the number of passengers for Wizz Air airline company. For this purpose, we will consider data from 2014 until 2026 that was collected month by month (available on Wizz Air website). Two methodologies will be considered. At first we will build a linear regression model and will show why this approach is inadequate as it breaks the assumptions of the linear regression model.
Data at first was splited into two time-groups. The first group consists of observations before 2026 - this will be our training set -  and the the second group consists of observations from 2026 and serves as the test set.
Underneath we can see a chart with our data from training set.


```{r message=FALSE, warning=FALSE, include=FALSE}
library(ggplot2)
library(dplyr)
library(lubridate)
library(forecast)
library(tseries)
library(modeltime)
library(timetk)
library(tidymodels)
library(Metrics)
library(knitr)
library(patchwork) 
library(lubridate)
library(broom)
```



```{r message=FALSE, warning=FALSE, include=FALSE}
data <- read.csv("C:\\Users\\mporc\\Downloads\\WizzAir_Statystyki_2014_20267.csv", sep = ";",header = TRUE)

months <- data.frame("Miesiąc" = c('Styczeń', "Luty", "Marzec", "Kwiecień", "Maj",
                                   'Czerwiec', "Lipiec", "Sierpień", "Wrzesień",
                                   'Październik', "Listopad", "Grudzień"),
                     "Number" = c(paste0(0, 1:9), 10, 11, 12))
```


```{r message=FALSE, warning=FALSE, include=FALSE}
data <- data %>%
  mutate(Liczba.Pasażerów = as.numeric(gsub(" ", "", Liczba.Pasażerów))) %>%
  left_join(months, by = "Miesiąc")%>%
  mutate(Data = my(paste(Number,"-", Rok))) %>%
  mutate(time = nrow(data):1)%>%
  select(Data, Liczba.Pasażerów, Covid_lockdown,	
         Covid_recovery1,	
         Covid_lockdown2,	
         Covid_recovery2,	
         Covid_wave3,	
         Covid_recovery3,	
         Post_covid, 
         time
  ) %>%
  arrange(Data) %>%
  rename(Pasazerowie = Liczba.Pasażerów) 

data_whole <- data
```





```{r include=FALSE}
data_test <- data %>%
  filter(Data >= "2025-07-01")

data <- data %>%
  filter(Data < "2025-07-01")
```

```{r out.width="75%", fig.align="center",echo=FALSE}
plot(data$Data, data$Pasazerowie, ylab = 'Passengers', xlab = 'Date', 
     yaxt = 'n', pch = 16)

labels <- c('0', "2 million", '4 million', '6 million')
poi <- c(0, 2000000, 4000000, 6000000)
axis(side = 2,at = poi, labels=labels)
```



The period of the Covid-19 outbreak deserves special attention. As we can see, from the Spring of 2020 till the end of 2021 we see a major decline in the number of Passengers. It is mainly connected with some restrictions at the airports, lockdown (especially in April and May 2020), Covid passports etc. In general travelling during this period was significantly restricted. If we got rid of that time interval, we would see that there is a rise in the number of Passengers year by year and that trend seems to be linear. Moreover, we could see that there are some months when the number of passengers is much higher comparing it to the other months. Especially in the summer when people go on holidays and a lot of people choose to go somewhere abroad. To picture that phenomenon, we can just look at the period from the beginning of 2023 to the summer 2025.

```{r echo=FALSE, fig.align="center", out.width="75%"}
time_series <- ts(data$Pasazerowie, start = c(2014,4), frequency = 12)
ggseasonplot(time_series, main = 'Number of passengers year by year')
```

Each year the number of Wizz Air customers reaches a peak in July or August since that is summer period. We can also notice a small rise in December, probably because of Christmas. 

We can now proceed to the construction of the first predictive model. We will use a linear regression model. It assumes that relation between explanatory variable and response variable is

\centering
$Y_i = \beta_0+\beta_1 X_{1i}+...+\beta_{p-1}X_{p-1 i} + \epsilon,$
  \raggedright

where epsilon is i.i.d random variable having normal distribution with mean zero and variance $\sigma^2$. In our case, we will build two models. First the first model, our explanatory variables will be dummy variables that consider only periods for Covid outbreak. We have seven periods of pandemic. The first one before Covid, second one considers April 2020 and May 2020 since that were two month of lockdown when airports were closed. We would not focus on March 2020 as the lockdown in many countries started in the middle of that month. Then we have a period of recovery after covid - generaly summer 2020. Then we faced covid second wave from Autumn 2020 to Spring 2021. Then was the next recovery - Summer 2021. The last phase of the outbreak was Covid third wave again mainly in Winter 2021 and then in Spring 2022 we have the last recovery after wave. From July 2022 we have post-Covid era when many restrictions concerning travelling were finished.




```{r echo=FALSE}
model_regresji_1 <- lm(Pasazerowie ~ Covid_lockdown+	
                         Covid_recovery1+	
                         Covid_lockdown2+	
                         Covid_recovery2+	
                         Covid_wave3+
                         Covid_recovery3+	
                         Post_covid, data = data)
knitr::kable(tidy(model_regresji_1), digits = 4)
```

Most of the coefficients in that model are important. Probably we could get rid of phases that are second and third recovery after Covid wave but it is not our main goal now.

We will also consider the second model that takes into consideration also time.



```{r echo=FALSE}
model_regresji_2 <- lm(Pasazerowie ~ Covid_lockdown+	
                         Covid_recovery1+	
                         Covid_lockdown2+	
                         Covid_recovery2+	
                         Covid_wave3+
                         Covid_recovery3+	
                         Post_covid+
                         time, data = data)
knitr::kable(tidy(model_regresji_2), digits = 4)
```

Looking at the R-squared coefficients we derive almost ideal model that perfectly fits our data. Again most of the coefficients are important. In this case, we could think of removing variable concerning post-Covid period.

But now we should also look at the residuals of those models to see if there are no problems with heteroscedasticity of that residuals are not independent.

```{r echo=FALSE, ,fig.cap="Residuals of two regression models",fig.pos = "H"}
par(mfrow = c(1,2))
plot(residuals(model_regresji_1), ylab = '', 
     pch = 16,
     xlab = '')
plot(residuals(model_regresji_2), ylab = '', 
     pch = 16,
     xlab = '')
par(mfrow = c(1,1))
```

Now we get to the point  that shows us why our two models are bad. They simply break assumptions of the linear model. In the first case we have problem with trend in our data so residuals are not independent. In the second case we can also notice a pattern concerning seasonality. In both cases we have also problem with heteroscedasticity so variance is not stable.

That is why we just cannot use a linear regression model for that purpose. We should take into consideration that we have a trend in our data and seasonality and that our residuals are correlated over time. That is why we should consider time series to solve this issue. But now we can face the other problem concerning Covid outbreak. Observations from that period can really have an influence on our predictions. Especially observations from first lockdown when almost nobody travelled by train. That is why we will decide on linear regression model with ARIMA errors. 

This model will help us with correlation among residuals and we can also use linear regression to consider different Covid outbreak periods. Hence, now we will consider a model


\centering
$Y_i = \beta_0+\beta_1 X_{1i}+...+\beta_{p-1}X_{p-1 i} + \eta_t,$
  \raggedright

where $\eta_t$ comes from the $ARIMA(p,d,q)(P,D,Q)_m$ model. We will be using regressors shown below to consider also Covid outbreak.

```{r echo=FALSE}
knitr::kable(tail(data) %>%
               rename(Lockdown1 = Covid_lockdown,	
                      Recovery1 =      Covid_recovery1,	
                      Lockdown2 = Covid_lockdown2,	
                      Recovery2 =  Covid_recovery2,	
                      Wave3 = Covid_wave3,	
                      Recovery3 = Covid_recovery3,	
                      Post_Covid = Post_covid))
```


Now let's look at our time series.

```{r include=FALSE}
time_series <- ts(data$Pasazerowie, start = c(2014,4), frequency = 12)

reg_matrix <- as.matrix(data%>% select(Covid_lockdown,	
                                       Covid_recovery1,	
                                       Covid_lockdown2,	
                                       Covid_recovery2,	
                                       Covid_wave3,	
                                       Covid_recovery3,	
                                       Post_covid))
```


```{r out.width="75%", fig.align="center",echo=FALSE,fig.pos = "H"}
plot(time_series, xlab = 'Date', ylab = 'Passengers', yaxt= 'n')
labels <- c('0', "2 million", '4 million', '6 million')
poi <- c(0, 2000000, 4000000, 6000000)
axis(side = 2,at = poi, labels=labels)
```

As it was stated before, we have a problem with trend in our data and seasonality. 

```{r out.width="75%", fig.align="center",echo=FALSE, message=FALSE, fig.pos = "H",warning=FALSE}
ggplot(data, aes(x=Data, y = Pasazerowie))+
  geom_line()+
  geom_smooth()
```
Now the main goal is to find the best ARIMA model for our residuals. We will try to build two models. The first will not include time in linear regression part and the second one will. We should now focus on finding hyperparameters $d, D$ that are responsible for differencing our model. We have a problem with trend and seasonality in our data so probably we should use $d =1$ and $D=1$ but to proof that we can look at the ACF and PACF functions. 

```{r out.width="75%", fig.align="center",echo=FALSE,fig.cap="ACF and PACF for ARIMA(0,0,0)(0,0,0)",fig.pos = "H"}
ggtsdisplay(Arima(time_series, order = c(0,0,0), seasonal = c(0,0,0), xreg = reg_matrix)$residuals)
```

As we can see we clearly have a problem with trend since on the graph with ACF we can notice some oscillations. There is also a bigger bar on the lag 12 so we have also a problem with seasonality.



```{r out.width="75%", fig.align="center",echo=FALSE, fig.cap="ACF and PACF for ARIMA(0,1,0)(0,1,0)",fig.pos = "H"}
ggtsdisplay(Arima(time_series, order = c(0,1,0), seasonal = c(0,1,0), xreg = reg_matrix)$residuals)
```

Now our model seems to be stationary but still we have a huge bars on lags 12 and 24 both on ACF and PACF graphs. To capture this we should now find coefficients $P$ and $Q$. As there is one (or two) huge bars on ACF functions and two huge bars on PACF functions we should mainly consider parameters $P,Q = 0,1,2$. Underneath we can see ACF and PACF graph functions with different $P$ and $Q$ parameters.

```{r out.width="75%", fig.align="center",echo=FALSE, fig.cap="ACF and PACF for ARIMA(0,1,0)(0,1,1)",fig.pos = "H"}
ggtsdisplay(Arima(time_series, order = c(0,1,0), seasonal = c(0,1,1), xreg = reg_matrix)$residuals)
```

```{r out.width="75%", fig.align="center",echo=FALSE,fig.cap="ACF and PACF for ARIMA(0,1,0)(0,1,2)",fig.pos = "H"}
ggtsdisplay(Arima(time_series, order = c(0,1,0), seasonal = c(0,1,2), xreg = reg_matrix)$residuals)
```

```{r out.width="75%", fig.align="center",echo=FALSE,fig.cap="ACF and PACF for ARIMA(0,1,0)(0,1,1)",fig.pos = "H"}
ggtsdisplay(Arima(time_series, order = c(0,1,0), seasonal = c(1,1,1), xreg = reg_matrix)$residuals)
```

```{r out.width="75%", fig.align="center",echo=FALSE,fig.cap="ACF and PACF for ARIMA(0,1,0)(1,1,0)",fig.pos = "H"}
ggtsdisplay(Arima(time_series, order = c(0,1,0), seasonal = c(1,1,0), xreg = reg_matrix)$residuals)
```

```{r out.width="75%", fig.align="center",echo=FALSE,fig.cap="ACF and PACF for ARIMA(0,1,0)(2,1,0)",fig.pos = "H"}
ggtsdisplay(Arima(time_series, order = c(0,1,0), seasonal = c(2,1,0), xreg = reg_matrix)$residuals)
```

In general the only combination of parameters $P$ and $Q$ that does not help us with seasonality is $P=1$ and $Q=0$. In other cases, bars at lags $12, 24, 36,...$ are below blue dotted line. There is only one bigger bar on lag 25 but it is acceptable for 5% for bars being above the dotted line so we will not focus on that much.

When it comes to decision about parameters $p$ and $q$ I do not think that they are necessary for the model because we do not notice any bigger bars at lags $1,2,$ etc. But we will also consider them because now we would like to find the best model basing on AIC metrics. The lesser AIC value for the model, the better. 


```{r include=FALSE}
params <- matrix(0, ncol = 6, nrow =20)
params[1,] <- c(0,1,0,0,1,1)
params[2,] <- c(1,1,0,0,1,1)
params[3,] <- c(0,1,1,0,1,1)
params[4,] <- c(1,1,1,0,1,1)
params[5,] <- c(0,1,0,0,1,2)
params[6,] <- c(1,1,0,0,1,2)
params[7,] <- c(0,1,1,0,1,2)
params[8,] <- c(1,1,1,0,1,2)
params[9,] <- c(0,1,0,1,1,1)
params[10,] <- c(1,1,0,1,1,1)
params[11,] <- c(0,1,1,1,1,1)
params[12,] <- c(1,1,1,1,1,1)
params[13,] <- c(0,1,0,2,1,1)
params[14,] <- c(1,1,0,2,1,1)
params[15,] <- c(0,1,1,2,1,1)
params[16,] <- c(1,1,1,2,1,1)
params[17,] <- c(0,1,0,2,1,0)
params[18,] <- c(1,1,0,2,1,0)
params[19,] <- c(0,1,1,2,1,0)
params[20,] <- c(1,1,1,2,1,0)

parameters <- apply(params, 1, function(x) {
  list('y' = time_series,
      'order' = c(x[1:3]),
      'seasonal' = c(x[4:6]),
      'xreg' = reg_matrix)
})

Arimas_models <- sapply(parameters, function(x){
   do.call(Arima,x)$aic
})

AIC_for_models <- data.frame("Model" = apply(params,1, function(x){
  paste('ARIMA(', paste(x[1:3],collapse=','), ')(', paste(x[4:6],collapse=','), ')')
}),
'AIC' = Arimas_models)
```

```{r echo=FALSE}
knitr::kable(AIC_for_models) 
```

As we can see, there are really small differences among AIC values between the models. But we have to find one best model and that is the model ARIMA(0,1,1)(0,1,2).

```{r echo=FALSE}
knitr::kable(AIC_for_models %>%
  filter(AIC == min(AIC)))
```

```{r out.width="75%", fig.align="center",echo=FALSE,fig.cap="ACF and PACF for ARIMA(0,1,1)(0,1,2)",fig.pos = "H"}
ggtsdisplay(Arima(time_series, order = c(0,1,1), seasonal = c(0,1,2), xreg = reg_matrix)$residuals)
```

ACF and PACF graphs are acceptable. We see bigger bars on lag 3 and 25 but there are no repetitive scenarios. 

We can also perform Ljung-Box test to test the correlation among residuals. For this test, null hypothesis states residuals behaves like white noise. Alternative hypothesis is the antithesis of null hypothesis. We will reject $H_0$ basing of $p$-value. If it is lesser than significance level $\alpha$, then we will reject null hypothesis. In other case, we will not.


```{r include=FALSE}
time_series_chosen <- Arima(time_series, order = c(0,1,1), seasonal = c(0,1,2), 
                            xreg = reg_matrix)
```

```{r echo=FALSE}
knitr::kable(tidy(Box.test(residuals(time_series_chosen), lag = 36, type = "Ljung-Box", fitdf = 1)))
```

In our case $p$-value is greater than significance level $\alpha$ (for that purpose we can use $\alpha = 0.05$) so we will not reject null hypothesis. That is a good sign.

Now we can also use other method. Now we will buid a second model that a time variable in the linear regression part.


```{r include=FALSE}
reg_matrix1 <- as.matrix(data%>% select(Covid_lockdown,	
                                       Covid_recovery1,	
                                       Covid_lockdown2,	
                                       Covid_recovery2,	
                                       Covid_wave3,	
                                       Covid_recovery3,	
                                       Post_covid,
                                       time))

```


```{r out.width="75%", fig.align="center",echo=FALSE,fig.cap="ACF and PACF for ARIMA(0,0,0)(0,0,0)",fig.pos = "H"}
ggtsdisplay(Arima(time_series, order = c(0,0,0), seasonal = c(0,0,0), xreg = reg_matrix1)$residuals)
```


We have a clear problem with seasonality. We have to do seasonal differencing, so $D=1$.


```{r out.width="75%", fig.align="center",echo=FALSE,fig.cap="ACF and PACF for ARIMA(0,0,0)(0,1,0)",fig.pos = "H"}
ggtsdisplay(Arima(time_series, order = c(0,0,0), seasonal = c(0,1,0), xreg = reg_matrix1)$residuals)
```


Still some huge bars at lags 12 on ACF graph function and PACF on lags 12 and 24. 

```{r out.width="75%", fig.align="center",echo=FALSE,fig.cap="ACF and PACF for ARIMA(0,0,0)(0,1,1)",fig.pos = "H"}
ggtsdisplay(Arima(time_series, order = c(0,0,0), seasonal = c(0,1,1), xreg = reg_matrix1)$residuals)
```

As we can see using parameter $Q=1$ and $P=0$ helps. There are other combinations that also help, i.e. $P=1$ and $Q=1$, $P=2$ and $Q=1$ etc. However, we still see huge bars on lag 1 on both ACF and PACF functions. That means that we should find suitable parameters $p$ and $q$ responsible for ARMA part in the model.

From PACF we can see that probably suitable should be $p=1$ because after a huge value on lag 1 then we see rapid decay on further lags. On ACF function we can see something similar. Maybe just parameter $p=1$ will be enough.

```{r out.width="75%", fig.align="center",echo=FALSE,fig.cap="ACF and PACF for ARIMA(1,0,0)(0,1,1)",fig.pos = "H"}
ggtsdisplay(Arima(time_series, order = c(1,0,0), seasonal = c(0,1,1), xreg = reg_matrix1)$residuals)
```

```{r out.width="75%", fig.align="center",echo=FALSE,fig.cap="ACF and PACF for ARIMA(0,0,1)(0,1,1)",fig.pos = "H"}
ggtsdisplay(Arima(time_series, order = c(0,0,1), seasonal = c(0,1,1), xreg = reg_matrix1)$residuals)
```

```{r out.width="75%", fig.align="center", echo=FALSE,fig.cap="ACF and PACF for ARIMA(1,0,1)(0,1,1)",fig.pos = "H"}
ggtsdisplay(Arima(time_series, order = c(1,0,1), seasonal = c(0,1,1), xreg = reg_matrix1)$residuals)
```

We see that a model with just $q=1$ is not suitable. The other shown models are acceptable. We also see that a model with parameter $q=2$ is much better than that with $q=1$

```{r out.width="75%", fig.align="center", echo=FALSE,fig.cap="ACF and PACF for ARIMA(0,0,2)(0,1,1)",fig.pos = "H"}
ggtsdisplay(Arima(time_series, order = c(0,0,2), seasonal = c(0,1,1), xreg = reg_matrix1)$residuals)
```

Now we will do the same thing as in the previous case so will find the best model basing on AIC value.


```{r include=FALSE}
params <- matrix(0, ncol = 6, nrow =9)
params[1,] <- c(1,0,0,0,1,1)
params[2,] <- c(1,0,1,0,1,1)
params[3,] <- c(0,0,2,0,1,1)
params[4,] <- c(1,0,0,1,1,0)
params[5,] <- c(1,0,1,1,1,0)
params[6,] <- c(0,0,2,1,1,0)
params[7,] <- c(1,0,0,1,1,1)
params[8,] <- c(1,0,1,1,1,1)
params[9,] <- c(0,0,2,1,1,1)

parameters <- apply(params, 1, function(x) {
  list('y' = time_series,
      'order' = c(x[1:3]),
      'seasonal' = c(x[4:6]),
      'xreg' = reg_matrix1)
})

Arimas_models_2 <- sapply(parameters, function(x){
   do.call(Arima,x)$aic
})

AIC_for_models_2 <- data.frame("Model" = apply(params,1, function(x){
  paste('ARIMA(', paste(x[1:3],collapse=','), ')(', paste(x[4:6],collapse=','), ')')
}),
'AIC' = Arimas_models_2)
```


```{r echo=FALSE}
knitr::kable(AIC_for_models_2)
```

The model with the lowest AIC is ARIMA(1,0,1)(1,1,1).

```{r echo=FALSE}
knitr::kable(AIC_for_models_2 %>%
  filter(AIC == min(AIC)))
```

The last thing to do is to check residuals of the model again by performing Ljung-Box test.

```{r include=FALSE}
time_series_chosen2 <- Arima(time_series, order = c(1,0,1), seasonal = c(1,1,1), 
                            xreg = reg_matrix1)
```

```{r echo=FALSE}
knitr::kable(tidy(Box.test(residuals(time_series_chosen2), lag = 36, type = "Ljung-Box", fitdf = 1)))
```
As we can see we do not reject a null hypothesis so there is nothing wrong with residuals of the model.

To compare those two models (as a reminder - ARIMA(0,1,1)(0,1,2) and ARIMA(1,0,1)(1,1,1) with time in linear regression part) we will have to base on their predictions. For that purpose, we will do a cross validation with expanding window. Assume that we split our training set into $k$ groups we respect to time. At first we train our data on the first, the oldest group and then test its results on the second group. Then we train our data on the first and second group and train it on the third group. And the whole proccess continues until we train our model on $k-1$ groups and test it on the last, the newest $k$-th group.

Underneath we can see how we splited our training set. 



```{r echo=FALSE}
resamples <- data %>%
  time_series_cv(
    date_var = Data,
    initial = '9 years',
    assess = '1 year',
    skip = '3 months',
    slice_limit = 5,
    cumulative = TRUE
  )
```

```{r out.width="75%", fig.align="center", echo=FALSE, fig.pos="H", fig.height=8}
library(ggplot2)
library(scales)

resamples %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(Data, Pasazerowie, .interactive = FALSE) +
  scale_y_continuous(labels = scales::label_number(scale_cut = cut_short_scale())) +
  theme(
    strip.text = element_text(size = 10, face = "bold"), 
    axis.text.y = element_text(size = 8)
  )
```

We have to have some metrics thanks to which we could compare the results between folds and two models. We will use to metrics - WAPE and MASE. WAPE metrics shows us overall forecast error as a simple percentage, telling us exactly how much your model missed the real total numbers. MASE metrics compares your model to a lazy, basic guess, where any score under 1.0 proves your advanced model is actually smarter and better.




```{r echo=FALSE}
tr_ts <- lapply(resamples$splits, function(x){ return(list(training(x), testing(x)))})
models_split <- lapply(tr_ts, function(x){ 
  t_series <- ts(x[[1]]$Pasazerowie, start = c(2014,4), frequency = 12)
  return(Arima(t_series, order = c(0,1,1), seasonal = c(0,1,2),
               xreg = as.matrix(x[[1]] %>%
                 select(colnames(reg_matrix)))))})

options(scipen=999)
wape_metric <- mapply(function(x,y) {
  predicted <- forecast(x, h=12, xreg = as.matrix(y[[2]]%>%                                                          select(colnames(reg_matrix))))$mean
  actual <- y[[2]]$Pasazerowie
  
  return(mape(actual, predicted))
  }, x = models_split, y = tr_ts)

mase_metric <- mapply(function(x,y){
  predicted <- forecast(x, h=12, xreg = as.matrix(y[[2]]%>%                                                          select(colnames(reg_matrix))))$mean
  actual <- y[[2]]$Pasazerowie
  train <- y[[1]]$Pasazerowie
  MASE_value <- mean(abs(actual-predicted))/mean(abs(diff(train, lag = 12)))
  return(MASE_value)
}, x = models_split, y = tr_ts)


cross_frame <- data.frame('Split' = 1:5, 'WAPE' = paste0(round(wape_metric,3)*100, '%'), 'MASE' = round(mase_metric,2))
                      
knitr::kable(cross_frame)

                                                    
```
The first model achieves really good results. MASE value in all cases is below value 1 so our model is in all cases better than naive model. WAPE model is also not bigger than 
16%, which is a percentage error that we make.

```{r echo=FALSE,fig.pos = "H", out.width = '85%', fig.align = 'center'}
plots <- mapply(function(x, y) {

  start_date <- min(y[[2]]$Data)

  ts1 <- ts(
    y[[2]]$Pasazerowie,
    start = c(year(start_date), month(start_date)),
    frequency = 12
  )

  prognoza <- forecast(
    x,
    h = 12,
    xreg = as.matrix(
      y[[2]] %>%
        select(colnames(reg_matrix))
    )
  )

  autoplot(prognoza) +
    autolayer(ts1, linewidth = 1) +
    autolayer(prognoza$mean, linewidth = 1, color = "black") +
    geom_line(linewidth = 1) +
    ggtitle(NULL) +
    theme(legend.position = "none")

},
x = models_split,
y = tr_ts,
SIMPLIFY = FALSE)



design <- "
AB
CD
E.
"

wrap_plots(
  A = plots[[1]],
  B = plots[[2]],
  C = plots[[3]],
  D = plots[[4]],
  E = plots[[5]],
  design = design
)
```

Also predictions include in 80 or 95% confidence interval which is a good sign. Now see the results obtained by the second model.



```{r echo=FALSE}
tr_ts <- lapply(resamples$splits, function(x){ return(list(training(x), testing(x)))})
models_split2 <- lapply(tr_ts, function(x){ 
  t_series <- ts(x[[1]]$Pasazerowie, start = c(2014,4), frequency = 12)
  return(Arima(t_series, order = c(1,0,1), seasonal = c(1,1,1),
               xreg = as.matrix(x[[1]] %>%
                 select(colnames(reg_matrix1)))))})

options(scipen=999)
wape_metric2 <- mapply(function(x,y) {
  predicted <- forecast(x, h=12, xreg = as.matrix(y[[2]]%>%                                                          select(colnames(reg_matrix1))))$mean
  actual <- y[[2]]$Pasazerowie
  
  return(mape(actual, predicted))
  }, x = models_split2, y = tr_ts)

mase_metric2 <- mapply(function(x,y){
  predicted <- forecast(x, h=12, xreg = as.matrix(y[[2]]%>%                                                          select(colnames(reg_matrix1))))$mean
  actual <- y[[2]]$Pasazerowie
  train <- y[[1]]$Pasazerowie
  MASE_value <- mean(abs(actual-predicted))/mean(abs(diff(train, lag = 12)))
  return(MASE_value)
}, x = models_split2, y = tr_ts)



cross_frame <- data.frame('Split' = 1:5, 'WAPE' = paste0(round(wape_metric2,3)*100, '%'), 'MASE' = round(mase_metric2,2))
                      
knitr::kable(cross_frame)

                                                    
```
This results are even better but we should remember that they were measured on the training set. We should remember that the final decision regarding which model should we choose will be made basing on the results obtained on the test set. We can see them below.


```{r, echo=FALSE, message=FALSE, warning=FALSE, out.width="85%", fig.align="center"}
plots <- mapply(function(x, y) {

  start_date <- min(y[[2]]$Data)

  ts1 <- ts(
    y[[2]]$Pasazerowie,
    start = c(year(start_date), month(start_date)),
    frequency = 12
  )

  prognoza <- forecast(
    x,
    h = 12,
    xreg = as.matrix(
      y[[2]] %>%
        select(colnames(reg_matrix1))
    )
  )

  autoplot(prognoza) +
    autolayer(ts1, linewidth = 1) +
    autolayer(prognoza$mean, linewidth = 1, color = "black") +
    geom_line(linewidth = 1) +
    ggtitle(NULL) +
    theme(legend.position = "none")

},
x = models_split2,
y = tr_ts,
SIMPLIFY = FALSE)

design <- "
AB
CD
E. 
"

wrap_plots(
  A = plots[[1]],
  B = plots[[2]],
  C = plots[[3]],
  D = plots[[4]],
  E = plots[[5]],
  design = design
)
```



```{r include=FALSE}
reg_future <- as.matrix(data_test%>% select(Covid_lockdown,	
                                       Covid_recovery1,	
                                       Covid_lockdown2,	
                                       Covid_recovery2,	
                                       Covid_wave3,	
                                       Covid_recovery3,	
                                       Post_covid))
```


```{r include=FALSE}
reg_future1 <- as.matrix(data_test%>% select(Covid_lockdown,	
                                       Covid_recovery1,	
                                       Covid_lockdown2,	
                                       Covid_recovery2,	
                                       Covid_wave3,	
                                       Covid_recovery3,	
                                       Post_covid,
                                       time))
```




```{r include=FALSE}
time_series_chosen <- Arima(time_series, order = c(0,1,1), seasonal = c(0,1,2), 
                            xreg = reg_matrix)
predicted <- forecast(time_series_chosen, h= 12, xreg= reg_future)$mean
actual <- data_test$Pasazerowie
mape1 <- mape(actual, predicted)
MASE_value1 <- mean(abs(actual-predicted))/mean(abs(diff(time_series, lag = 12)), na.rm = TRUE)

```

```{r include=FALSE}
time_series_chosen2 <- Arima(time_series, order = c(1,0,1), seasonal = c(1,1,1), 
                            xreg = reg_matrix1)
predicted <- forecast(time_series_chosen2, h= 12, xreg= reg_future1)$mean
actual <- data_test$Pasazerowie
mape2 <- mape(actual, predicted)
MASE_value2 <- mean(abs(actual-predicted))/mean(abs(diff(time_series, lag = 12)), na.rm = TRUE)
```

```{r echo=FALSE}
final_results <- data.frame('Model' = c('ARIMA(0,1,1)(0,1,2)', 'ARIMA(1,0,1)(1,1,1) with time in regression part'), 
                            'AIC' = round(c(min(AIC_for_models$AIC), min(AIC_for_models_2$AIC)),2),
                            'WAPE' = c(paste0(round(mape1, 3)*100,'%'), paste0(round(mape2, 3)*100, '%')),
                            'MASE' = c(round(MASE_value1, 2), round(MASE_value2,2)))
knitr::kable(final_results)
```
As we can see, the first model has slightly better results. The second model seems to have a problem with overfitting as it had MASE value not greater than 0.3 but on the test set it is more than 0.5. It is still okay, could be worse, but it can be a little problem. First model seems to generalize better.

We can also see below how our forecast looks at the graphs.


```{r out.width="75%", fig.align="center",echo=FALSE,fig.pos = "H"}
autoplot(forecast(Arima(time_series, order = c(0,1,1), seasonal = c(0,1,2), xreg=reg_matrix), h=12, xreg = reg_future))+
  autolayer(ts(data_test$Pasazerowie, start = c(2025,7), frequency = 12))+
  theme(legend.position = "none")
```

```{r out.width="75%", fig.align="center",echo=FALSE,fig.pos = "H"}
autoplot(forecast(Arima(time_series, order = c(1,0,1), seasonal = c(1,1,1), xreg=reg_matrix1), h=12, xreg = reg_future1))+
  autolayer(ts(data_test$Pasazerowie, start = c(2025,7), frequency = 12))+
  theme(legend.position = "none")
```

I would say that the second model underestimated the rise in the number of passengers in Spring 2026 and that is why it loses with the first model.

At the end, we will just try to use the chosen model and use it to make some predictions about the future. As we can see, our model estimates that in the following year the number of customers will be close to 9 million in July. Even this year in July the number of Wizz Air airlines customers is expected to reach 8 million easily.

```{r}
time_forecast <- ts(data_whole$Pasazerowie, start = c(2014,4), frequency = 12)
reg_forecast <- as.matrix(data_whole %>% select(
                                      Covid_lockdown,	
                                       Covid_recovery1,	
                                       Covid_lockdown2,	
                                       Covid_recovery2,	
                                       Covid_wave3,	
                                       Covid_recovery3,	
                                       Post_covid
  
))


model_forecast <- Arima(time_forecast, order = c(0,1,1), seasonal = c(0,1,2),
                        xreg = reg_forecast)

reg_forecast2027 <- as.matrix(tail(data_whole %>% select(
                                      Covid_lockdown,	
                                       Covid_recovery1,	
                                       Covid_lockdown2,	
                                       Covid_recovery2,	
                                       Covid_wave3,	
                                       Covid_recovery3,	
                                       Post_covid
), 12))

prediction_2027 <- forecast(model_forecast, xreg= reg_forecast2027, h = 12)

```

```{r}
autoplot(prediction_2027)+
  ggtitle('Forecast for the number of Wizz Air passengers from July 2026 to July 2027')
```

To sum up, we obtained an ARIMA model thanks to which we can forecast the number of passengers of Wizz Air airlines in the future. We compared a lot of model, found the best basing on AIC value. We also made a cross validation with expanding window and measure MASE and WAPE metrics for our predictions.Finally, based on the results obtained on the test set, we were able to select the best model for this forecasting problem - $ARIMA(0,1,1)(0,1,2)_{12}$. At the end we made some forecast for the number of Wizz Air passengers in the following twelve months.