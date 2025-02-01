# Predicting power consumption for electric cars by Watt The Data

## General data information
We have a dataset containing information on approximately 100,000 electric car charging sessions in Norway. The data spans from January 2022 to June 2024 and focuses on sessions using fast chargers by Eveny. The chargers in use have nominal power ratings ranging from 50 DC to 400 DC, providing a diverse set of data points for analysis.

![Time](visualizations/time.png)

![Nominal Power Dist](visualizations/nominal%20power%20distribution.png)

A general trend we see is that the charging power reduces when the SOC is increasing. Specifically around when reaching the 80% state of charge level.
![SOC vs Power](visualizations/SOC%20vs%20Power%20trend.png)

For all the different nominal powers, the power consumption to the car follows the same pattern that it slowly declines over time.
![Power for each nominal power](visualizations/power%20trend%20for%20different%20nominal%20powers.png)

When we also include the State of Charge trend we can see the same pattern as earlier. 
![SOC vs Power for each nominal](visualizations/SOC%20vs%20power%20for%20each%20nominal%20power.png)

The power is also greatly impacted by the month of the charging. Likely because of weather conditions.
![Power each season](visualizations/median%20power%20each%20season.png)

Specifically we see that the colder seasons have a slower charging rate. When we look at each month seperately we can see that some month correlate highly together. The colder the month, the longer time the charger uses to reach optimal power. January and December are the slowest as they are the coldest months. November, February and March also have similar powar as they have roughly the same temperature. The months June, July and August have the fastest time to reaching optimal charging power.
![Power each month](visualizations/median%20power%20each%20month.png)


## Data preperation
We have set each id to be a row. So each id is a charging sessions of 40 minutes. We also have a charging session with an invalid State Of Charge above 100%, specifically 104%. This oberservation has therefor been removed. We also have some power data that has power over nominal power of that charging station. Some have minimal differences ex. 0.1/0.2, we have decided to keep as we believe minimal differences won't impact model performace. But we have a lot of misreadings on location_id 6. There we have powers from 500 to 2400 with a nominal charging of 350. None of the session iDs has multiple of these invalid reading. We therefor set only that observation to NaN for later imputation. We have decided to set a treshold that if power is 1 over the nominal powers we discard that observation, and do later imputation if the other observations for that charging session is correct.
![Inconsitencies nominal power](visualizations/inconsitencies%20location%20IDs.png)


## Missing values
State of Charge (SOC) and Power are the only variables that have missing values.
![Missing Power Values](visualizations/Missing%20Power%20Values.png)
![Missing SOC Values](visualizations/Missing%20Soc%20Values.png)

We have some charging sessions that have a lot of missing data. For simplicity for imputation we have removed charging sessions that has more than ... missing values. 


