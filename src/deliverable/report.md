# Predicting power consumption for electric cars by Watt The Data

## General data information
We have a dataset containing information on approximately 100,000 electric car charging sessions in Norway. The data spans from January 2022 to June 2024 and focuses on sessions using fast chargers by Eveny. The chargers in use have nominal power ratings ranging from 50 DC to 400 DC, providing a diverse set of data points for analysis.

![Time](visualizations/time.png)

Overview of the number of charging sessiong over datasets time span, showing a general increasing trend. The figure illustrates how charging activity has changed over time, potentially indicating shifts in demand, seasonality, or other influencing factors.

## legg til kake diagram

The pie chart illustrates the distribution of nominal power categories across the dataset.

![SOC vs Power](visualizations/SOC%20vs%20Power%20trend.png)

The line plot visualizes the relationship between state of charge (SoC) and charging power over time. 
A general trend emerges, showing a linear increase in SoC while power gradually declines. Notably, the charging power reduction becomes more pronounced as SoC approaches 80%.

![Power each season](visualizations/median%20power%20each%20season.png)

This line plot displays the median power at each minute, categorized by season.  We observe that winter starts at the lowest power level and follows a flatter curve, while summer begins at the highest and declines more steeply over time. Interestingly, spring and autumn exhibit nearly identical curves, suggesting similar charging power behavior in these transitional seasons. Likely because of weather conditions.

![Power each month](visualizations/median%20power%20each%20month.png)

To investigate the impact of seasonality on power curves, we visualize average power by month in a line plot, colored by average temperature. A clear seasonal trend emerges, where warmer months start with higher power and gradually decline throughout the session. In contrast, colder months begin with lower power, increase to a peak, and then decrease toward the end.

### legge til normal fordeling

This histogram shows total power consumption by the hour of the day. We observe an aggregated consumption pattern peaking around midday, forming a distribution resembling a normal curve.


## Data preperation
We have set each id to be a row. So each id is a charging sessions of 40 minutes. We also have a charging session with an invalid State Of Charge above 100%, specifically 104%. This oberservation has therefor been removed. We also have some power data that has power over nominal power of that charging station. Some have minimal differences ex. 0.1/0.2, we have decided to keep as we believe minimal differences won't impact model performace. But we have a lot of misreadings on location_id 6. There we have powers from 500 to 2400 with a nominal charging of 350. None of the session iDs has multiple of these invalid reading. We therefor set only that observation to NaN for later imputation. We have decided to set a treshold that if power is 1 over the nominal powers we discard that observation, and do later imputation if the other observations for that charging session is correct.
![Inconsitencies nominal power](visualizations/inconsitencies%20location%20IDs.png)


## Missing values
State of Charge (SOC) and Power are the only variables that have missing values.
![Missing Power Values](visualizations/Missing%20Power%20Values.png)
![Missing SOC Values](visualizations/Missing%20Soc%20Values.png)

We have some charging sessions that have a lot of missing data. For simplicity for imputation we have removed charging sessions that has more than ... missing values. 


