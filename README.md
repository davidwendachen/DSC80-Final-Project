# Predicting Outage Duration with Regression Models

## Final Project for DSC80 - UCSD
By David Chen

# Introduction

In this project, I analyzed a power outage dataset, which covers all instances of power outage from January 2000 to July 2016 in the United States. This dataset is from Purdue University’s Laboratory for Advancing Sustainable Critical Infrastructure (https://engineering.purdue.edu/LASCI/research-data/outages). 

The dataset details power outage data, such as outage duration, cause of outage, outage start time. It also contains other relevant information including geographical data, climate data, electric consumption data, etc. 

The question I will be tackling in my analysis will be what are the most critical predictors for outage duration, that is to say, which variables have the largest effect on outage duration length. To do this, I will be building a regression model that finds the variables with the most impact on outage duration. This will be very important for the energy companies, as it allows them to understand which factors are more responsible for longer outages, and can help them prioritize mitigation efforts.

The original DataFrame consists of 1534 rows, each corresponding to a specific outage, and 57 columns, each correpsonding to a variable. I will only be focusing on a few variables for my analysis:

<table border="1">
  <thead>
    <tr>
      <th>Variable</th>
      <th>Description</th>
      <th>Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>YEAR</td>
      <td>The year an outage occurred</td>
      <td>Numeric</td>
    </tr>
    <tr>
      <td>MONTH</td>
      <td>The month an outage occurred</td>
      <td>Temporal</td>
    </tr>
    <tr>
      <td>US.STATE</td>
      <td>The State an outage occurred in</td>
      <td>Categorical</td>
    </tr>
    <tr>
      <td>NERC.REGION</td>
      <td>North American Electric Reliability Corporation (NERC) region involved in the outage event</td>
      <td>Categorical</td>
    </tr>
    <tr>
      <td>CLIMATE.REGION</td>
      <td>U.S. Climate regions as specified by National Centers for Environmental Information the outage occured in</td>
      <td>Categorical</td>
    </tr>
    <tr>
      <td>CAUSE.CATEGORY</td>
      <td>Categories of all events causing outages</td>
      <td>Categorical</td>
    </tr>
    <tr>
      <td>CAUSE.CATEGORY.DETAIL</td>
      <td>Detailed description of the event categories causing the major power outages</td>
      <td>Categorical</td>
    </tr>
    <tr>
      <td>OUTAGE.DURATION</td>
      <td>Total length of the power outage (minutes)</td>
      <td>Numeric</td>
    </tr>
    <tr>
      <td>OUTAGE.START.DATE</td>
      <td>The start date of the power outage</td>
      <td>Temporal</td>
    </tr>
    <tr>
      <td>OUTAGE.START.TIME</td>
      <td>The start time of the power outage</td>
      <td>Temporal</td>
    </tr>
    <tr>
      <td>OUTAGE.RESTORATION.DATE</td>
      <td>The end date of the power outage</td>
      <td>Temporal</td>
    </tr>
    <tr>
      <td>OUTAGE.RESTORATION.TIME</td>
      <td>The end time of the power outage</td>
      <td>Temporal</td>
    </tr>
    <tr>
      <td>DEMAND.LOSS.MW</td>
      <td>Megawatts of demand lost during the outage</td>
      <td>Numeric</td>
    </tr>
    <tr>
      <td>CUSTOMERS.AFFECTED</td>
      <td>Number of customers affected by the outage</td>
      <td>Numeric</td>
    </tr>
    <tr>
      <td>TOTAL.CUSTOMERS</td>
      <td>Total number of customers served in the U.S. State the outage occured in</td>
      <td>Numeric</td>
    </tr>
    <tr>
      <td>POPPCT_URBAN</td>
      <td>Percentage of the total population of the U.S. state represented by the urban population (in %)</td>
      <td>Numeric</td>
    </tr>
    <tr>
      <td>POPDEN_URBAN</td>
      <td>Population density of the urban areas (persons per square mile)</td>
      <td>Numeric</td>
    </tr>
    <tr>
      <td>AREAPCT_URBAN</td>
      <td>Percentage of the land area of the U.S. state represented by the land area of the urban areas (in %)</td>
      <td>Numeric</td>
    </tr>
    <tr>
      <td>ANOMALY.LEVEL</td>
      <td>3 month running mean of the Oceanic El Niño/La Niña (ONI) index referring to the cold and warm episodes by season.</td>
      <td>Numeric</td>
    </tr>
    <tr>
      <td>PC.REALGSP.STATE</td>
      <td>Per capita real gross state product (GSP) in the U.S. state.</td>
      <td>Numeric</td>
    </tr>
  </tbody>
</table>

# Data Cleaning

First, I dropped all columns not used in my analysis, and kept only the variables from above, which are <code>YEAR</code>, <code>MONTH</code>, <code>US.STATE</code>, <code>NERC.REGION</code>, <code>CLIMATE.REGION</code>, <code>CAUSE.CATEGORY</code>, <code>CAUSE.CATEGORY.DETAIL</code>, <code>OUTAGE.DURATION</code>, <code>OUTAGE.START.DATE</code>, <code>OUTAGE.START.TIME</code>, <code>OUTAGE.RESTORATION.DATE</code>, <code>OUTAGE.RESTORATION.TIME</code>, <code>DEMAND.LOSS.MW</code>, <code>CUSTOMERS.AFFECTED</code>, <code>TOTAL.CUSTOMERS</code>

Next, we combine <code>OUTAGE.START.DATE</code> and <code>OUTAGE.START.TIME</code> columns into <code>OUTAGE.START</code>, and simiarly for <code>OUTAGE.RESTORATION.DATE</code> and <code>OUTAGE.RESTORATION.TIME</code> into <code>OUTAGE.RESTORATION</code>. Essentially we condensed two columns describing time into one single column. This helps ease our analysis of these variables later on.

I also replaced all values of 0 in <code>OUTAGE.DURATION</code>, <code>DEMAND.LOSS.MW</code>, and <code>CUSTOMERS.AFFECTED</code> with nan, since it is fair to assume that values of 0 for these columns is likely due to missingness. This will help ease our missingness analysis later on. 

The first rew rows of the dataframe is shown below.

<div style="overflow-x: auto; white-space: pre;">
<pre>
|   YEAR |   MONTH | U.S._STATE   | NERC.REGION   | CLIMATE.REGION     | CAUSE.CATEGORY     | CAUSE.CATEGORY.DETAIL   |   OUTAGE.DURATION | OUTAGE.START        | OUTAGE.RESTORATION   |   DEMAND.LOSS.MW |   CUSTOMERS.AFFECTED |   TOTAL.CUSTOMERS |   POPPCT_URBAN |   POPDEN_URBAN |   AREAPCT_URBAN |   ANOMALY.LEVEL |   PC.REALGSP.STATE |
|-------:|--------:|:-------------|:--------------|:-------------------|:-------------------|:------------------------|------------------:|:--------------------|:---------------------|-----------------:|---------------------:|------------------:|---------------:|---------------:|----------------:|----------------:|-------------------:|
|   2011 |       7 | Minnesota    | MRO           | East North Central | severe weather     | nan                     |              3060 | 2011-07-01 17:00:00 | 2011-07-03 20:00:00  |              nan |                70000 |           2595696 |          73.27 |           2279 |            2.14 |            -0.3 |              51268 |
|   2014 |       5 | Minnesota    | MRO           | East North Central | intentional attack | vandalism               |                 1 | 2014-05-11 18:38:00 | 2014-05-11 18:39:00  |              nan |                  nan |           2640737 |          73.27 |           2279 |            2.14 |            -0.1 |              53499 |
|   2010 |      10 | Minnesota    | MRO           | East North Central | severe weather     | heavy wind              |              3000 | 2010-10-26 20:00:00 | 2010-10-28 22:00:00  |              nan |                70000 |           2586905 |          73.27 |           2279 |            2.14 |            -1.5 |              50447 |
|   2012 |       6 | Minnesota    | MRO           | East North Central | severe weather     | thunderstorm            |              2550 | 2012-06-19 04:30:00 | 2012-06-20 23:00:00  |              nan |                68200 |           2606813 |          73.27 |           2279 |            2.14 |            -0.1 |              51598 |
|   2015 |       7 | Minnesota    | MRO           | East North Central | severe weather     | nan                     |              1740 | 2015-07-18 02:00:00 | 2015-07-19 07:00:00  |              250 |               250000 |           2673531 |          73.27 |           2279 |            2.14 |             1.2 |              54431 |
</pre>
</div>

# Exploratory Data Analysis

## Univariate Analysis

<iframe src="assets/figure_1.html" style="width:100%; height:450px; border:0;"></iframe>

Figure 1 depicts the distribution of binned outage durations. The plot shows that as each day passes, the number of outage still not resolved decrease significantly. This logarithmic relationship shows that extended outages are rare compared to shorter ones.
<iframe src="assets/figure_2.html" style="width:100%; height:450px; border:0;"></iframe>

Figure 2 shows the distribution of the cause categories. The most populated categories are <b>severe weather</b> and <b>intentional attack</b>, while the least populated categories are <b>fuel supply emergency</b> and <b>islanding</b>

<iframe src="assets/figure_3.html" style="width:100%; height:450px; border:0;"></iframe>

Figure 3 shows the number of outages with a detailed cause for each cause category. Unsurprisingly <b>severe weather</b> has the highest count, but weirdly enough <b>intentional attack</b> has one of the lowest, despite being one of the most populated categories from Figure 2. This is likely due to the fact that intentional attacks are a quite straightforward cause, meaning there is not much need to add additional details.

<iframe src="assets/figure_4.html" style="width:100%; height:450px; border:0;"></iframe>

Figure 4 shows the comparison between raw demand loss and log transformed demand loss. The raw demand loss values are extremely right-skewed, with most outages involving very small demand loss and a few rare events reaching high loss. After applying a log transformation, the distribution becomes much more balanced and approximately bell-shaped.

## Bivariate Analysis

<iframe src="assets/figure_5.html" style="width:100%; height:450px; border:0;"></iframe>

Figure 5 depicts the relationship between cause category and outage duration. Notice that outage durations vary widely across cause categories, with severe weather showing both the highest spread and some of the longest outages overall. Categories like intentional attack, equipment failure, and system operability disruption tend to have shorter and more consistent outage durations, reflected by their much tighter boxplots. Meanwhile, fuel supply emergencies exhibit several extreme long-duration outliers, suggesting that despite being rare, these events can lead to prolonged outages.

## Aggregates

| CAUSE.CATEGORY                |   avg_duration |   median_duration |   count |
|:------------------------------|---------------:|------------------:|--------:|
| fuel supply emergency         |        13484.00 |            3960.0 |      38 |
| severe weather                |         3899.71 |            2464.0 |     741 |
| equipment failure             |         1850.56 |             224.0 |      54 |
| public appeal                 |         1468.45 |             455.0 |      69 |
| system operability disruption |          747.09 |             222.0 |     120 |
| intentional attack            |          521.93 |              92.5 |     332 |
| islanding                     |          200.55 |              77.5 |      44 |

I grouped on the cause categories and found the mean, median, and count for each category. Notice that <b>fuel supply emergency</b> and <b>severe weather</b> has the highest averages (both median and mean), while <b>intentional attacks</b> and <b>islanding</b> has the lowest. 

# Assessment of Missingness

## NMAR Analysis

In the dataset, one variable that could be NMAR is <code>CUSTOMERS.AFFECTED</code>. I believe that whether a utility reports this value likely depends on how many customers were actually affected. For example, they may be less inclined to record very small or very large events for various reasons. 

## Missingness Dependency

I tested missingness dependecy for the <code>OUTAGE.DURATION</code> column. I looked at the distribution of <code>CAUSE.CATEGORY</code> and <code>MONTH</code> when <code>OUTAGE.DURATION</code> is missing vs not missing. 

To do this, I performed permutation chi-square tests comparing the <code>DURATION_MISSING</code>, a missingness indicator, to both <code>CAUSE.CATEGORY</code> and <code>MONTH</code>

### CAUSE.CATEGORY

The hypotheses are as follows:
Null Hypothesis: Missingness of <code>OUTAGE.DURATION</code>is independent of <code>CAUSE.CATEGORY</code>; any differences in missingness rates across causes are due to random chance.
Alternative Hypothesis: Missingness of <code>OUTAGE.DURATION</code> depends on <code>CAUSE.CATEGORY</code>; at least one cause category has a different probability of missing duration than the others.

<iframe src="assets/figure_6.html" style="width:100%; height:450px; border:0;"></iframe>

For <code>CAUSE.CATEGORY</code>, the observed chi-square statistic is extremely large with a value of around 131.9, and a p-value of essentially 0. With this, I reject the null hypothesis, and accept that missing outage durations likely depends on the observation’s cause category.

### MONTH

The hypotheses are as follows:
Null Hypothesis: Missingness of <code>OUTAGE.DURATION</code> is independent of <code>MONTH</code>; any differences in missingness across months is due to random chance.
Alternative Hypothesis: Missingness of <code>OUTAGE.DURATION</code> depends on <code>MONTH</code>; at least one month has a different probability of missing duration than the others.

<iframe src="assets/figure_7.html" style="width:100%; height:450px; border:0;"></iframe>

For <code>MONTH</code>, the observed chi-square statistic is around 13.74, and the p-value is 0.239. With this, I fail to reject the null hypothesis, and acknowledge that the month of the year is likely not affected when outage duration is missing. 

# Hypothesis Testing

For my hypothesis tests, I will be testing to see if <code>OUTAGE.DURATION</code> is distributed the same across all cause categories. 

Null Hypothesis: The mean outage duration is the same across all cause categories, and any differences are due to random variation. 
Alternate Hypothesis: At least one cause category has a different mean outage duration, indicating that outage duration depends on the outage cause and differences are not due to random variation.

I will be performing a permutation test using a one-way ANOVA F-statistic as the test statistic and a significance level of 0.05.

<iframe src="assets/figure_8.html" style="width:100%; height:450px; border:0;"></iframe>

From Figure 8, we see that the observed F value is around 42.17, and the p-value is essentially 0. With this, I reject the null hypothesis at the 5% significance level. I believe that with this there is strong evidence that outage duration does depend on cause category.

# Prediction Problem

The prediction problem I plan on tackling is the prediction of the <code>OUTAGE.DURATION</code> variable. This will be a regression problem, as I am attempting to regress <code>OUTAGE.DURATION</code> on a set of predictors, made of columns I selected in the Introduction. I chose <code>OUTAGE.DURATION</code> because a better understanding of which factors drive longer outages can help utilities prioritize mitigation efforts.

The metric I used was MAE (mean absolute error), chosen for its interpretability and its relative robustness to outliers. I preferred MAE over RMSE (root mean squared error), since RMSE penalizes large errors more heavily and would be overly influenced by the outliers. As seen in the EDA section, <code>OUTAGE.DURATION</code> is highly skewed rather than bell-shaped, so using MAE helps prevent these outliers from dominating the evaluation.

It is worth to note that some columns in the dataset are not known at the time of prediction, such as <code>CUSTOMERS.AFFECTED</code> and <code>CAUSE.CATEGORY</code>, so the regression model will not be using these variables. The exact variables I will be using will be elaborated more in the Final Model section. 

# Baseline Model

The baseline model I used is a regression model that predicts <code>OUTAGE.DURATION</code> using <code>DEMAND.LOSS.MW</code> (quantitative), <code>YEAR</code> (temporal), and <code>CLIMATE.REGION</code> (nominal). The numeric columns (<code>DEMAND.LOSS.MW</code> and <code>YEAR</code>) are median-imputed and standardized, while the categorical column (<code>CLIMATE.REGION</code>) is imputed with the most frequent value and one-hot encoded. 

LinearRegression is used to learn the best-fit relationship between these transformed columns and <code>OUTAGE.DURATION</code>, then evaluated with MAE on a held-out test set.

This model achieved a test MAE of 2812.73, which is quite large and can definitely be improved on. 

# Final Model

For the final model, I chose the following features for my model:
<ul>
  <li><code>DEMAND.LOSS.MW</code></li>
  <li><code>TOTAL.CUSTOMERS</code> </li>
  <li><code>POPPCT_URBAN</code></li>
  <li><code>POPDEN_URBAN</code></li>
  <li><code>AREAPCT_URBAN</code></li>
  <li><code>YEAR</code></li>
  <li><code>MONTH</code></li>
  <li><code>U.S._STATE</code></li>
  <li><code>OUTAGE.START.HOUR</code></li>
  <li><code>CLIMATE.REGION</code></li>
  <li><code>NERC.REGION</code></li>
  <li><code>ANOMALY.LEVEL</code></li>
  <li><code>PC.REALGSP.STATE</code></li>
</ul>

<code>DEMAND.LOSS.MW</code> was chosen since larger demand loss indicates a more severe disruption, which I thought could be linked to outage duration. <code>TOTAL.CUSTOMERS</code> was chosen since regions with more total customers could have more complex power grids, which is correlated to restoration length and outage duration. The urban indicators <code>POPPCT_URBAN</code>, <code>POPDEN_URBAN</code>, and <code>AREAPCT_URBAN</code> were included because urban areas typically have higher power usage, indicating a more complex restoration process, which has an effect on outage duration. <code>YEAR</code> was selected since restoration processes likely changes over time, and due to the wide range of time covered by the dataset, the year the outage occurred in could have an impact on its duration. <code>MONTH</code> and <code>OUTAGE.START.HOUR</code> were chosen since restoration processes or response time could be different depending on the season of the month or the time of the day. <code>U.S._STATE</code> and <code>NERC.REGION</code> were included due to different states and regions having different power grid infrastructures. <code>CLIMATE.REGION</code> and <code>ANOMALY.LEVEL</code> were selected due to the large impact climate has on power grids, as seen by the category <b>severe weather</b> in <code>CAUSE.CATEGORY</code>. <code>PC.REALGSP.STATE</code> was chosen due to the potential correlation between state economic strength and power grid infrastructure. 

A Random Forest Regression model was chosen for its ability to model nonlinear relationships and interactions between predictors that are unlikely to be captured by a linear model. I tried 3 different values for each hyperparameter I used, which includ the number of trees (100, 300, 500), tree depth (none, 10, 20), and minimum leaf size (1, 5, 10). They were selected using 5-fold GridSearchCV, optimizing MAE to balance bias and variance.

The final model achieved a test MAE of 2495.96, which is significantly lower than that of the baseline model, showing an improved predictive performance.

# Fairness Analysis

For the fairness analysis, I defined Group X as outages with high demand loss and Group Y as outages with low demand loss, where demand loss was binarized using the median of <code>DEMAND.LOSS.MW</code>. The evaluation metric I chose was RMSE. 

Null Hypothesis: the model is fair with respect to demand loss, the RMSE for high-demand and low-demand outages is approximately equal, and any observed difference is due to random variation.
Alternative Hypothesis: the model is unfair with respect to demand loss, the RMSE differs between high-demand and low-demand outages.
The test statistic was the difference in RMSE (high − low), and a permutation test with 5,000 resamples was used at a significance level of 0.05 to approximate the null distribution.

<iframe src="assets/figure_9.html" style="width:100%; height:450px; border:0;"></iframe>

The observed RMSE difference is -1451.86, and the resulting p-value is 0.5602, which is not statistically significant. This shows that there is insufficient evidence to reject the null hypothesis, and we do not find strong evidence that the model performs worse for outages with higher demand loss.
