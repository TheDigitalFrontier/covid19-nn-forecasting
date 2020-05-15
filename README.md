# <img style="float: left; padding-right: 10px; width: 45px" src="https://raw.githubusercontent.com/Harvard-IACS/2018-CS109A/master/content/styles/iacs.png"> Modeling COVID-19 Growth with Neural Networks
## AC209B: Advanced Topics in Data Science



**Harvard University**<br/>
**Spring 2020**<br/>
**Team members**: William Seaton, Johannes Kolberg, Hardik Gupta


<hr style="height:2pt">

**Summary**: Accurate understanding of the severity and spread of the novel coronavirus COVID-19 is critical to implementing an effective public health response. As with any novel virus, COVID-19 has unique characteristics and data challenges that make accurate modeling difficult and there has been much public debate about the correct methodology and resulting output that we should use to guide our decisions. We propose various methods based on the architecture of Recurrent Neural Networks and discuss their accuracy and usability for predicting COVID-19 case growth at various time lengths.

----

There has been consistent speculation about the accuracy of coronavirus case counts around the world. Different countries have varying testing capacity and requirements, resulting in a patchwork of testing rates and coverage. In the United States, academics and news organizations have reported that the actual coronavirus case count is being underreported.

Some countries benefited from advanced notice as they were impacted by the spread of COVID-19 later than others. These countries in particular can use the growth rates of countries ahead of them to learn from and model their own expected growth rates.

Public health goals demand pandemic forecasting provide three things:

1) Accurate number of case growth

2) Predicted as far into the future as confidence will allow

3) Delivered as early as sufficient data is available

For the methods we use, we explore their capacity along each of these dimensions.

COVID-19 has a few significant modeling challenges. First, the spread of coronavirus is dependent on innate national qualities such as population size and geographic density as well as situational factors such as public health infrastructure and policy choices. This can lead to very different growth curves across countries. Second, because COVID-19 has unique characteristics, like infection rate and severity, it must be modeled separately from other viruses. This means we must try to produce models and predictions as early as possible with extremely limited data. As the pandemic spreads, we will gain additional data that can be used to refine our predictions but our goal should be to produce useable numbers that guide policy decisions early. Finally, case counts are adjusted retroactively on several occasions as the policy for counting cases changes, resulting in jagged growth curves that would likely look a lot more smoother if consistent measuring schemes had been used from the beginning. This presents additional challenges for the models, as they need to anticipate when cases are underreported and will subsequently "catch up".

We propose applying neural networks to modeling COVID-19 because of unique advantages in the underlying architecture in overcoming these challenges. In particular, we rely heavily on Recurrent Neural Networks (RNNs), as these are designed specificially for sequential data, which includes time series such as ours. RNNs allow stateful predictions so we can update our model as new data comes in and prioritize the latest numbers, especially important when dealing with exponential growth. RNNs can learn the shapes of multiple input time series when modeling expected number of cases, allowing data scientists to use related time series that may be easier and more accurate to collect. RNNs provide flexibility in architecture definition to deal with very different but related time series, like cumulative cases or new cases per day.

We apply RNNs and associated models to three hypotheses:

1) Are there general infection curve development patterns for COVID-19 across countries?

2) Can alternate time series assist with forecasting COVID-19 cases?

3) Can COVID-19 growth in other countries assist with forecasting cases in a target nation?

---------------


#### Import necessary packages


```python
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Flatten, Dropout, Bidirectional, TimeDistributed, Conv1D, MaxPool1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Import class for reservoir computing
from pyESN import ESN # Written by user cknd: https://github.com/cknd/pyESN

import warnings
warnings.filterwarnings('ignore')
```

# Data Preparation

We incorporate multiple time series to aid in forecasting various response variables related to COVID-19 spread, such as cumulative case count and new cases per day. Our time series include case and death count from the ECDC, test counts for several countries as collected by the website OurWorldInData and population mobility data provided separately by Apple and Google. Our experiments assess which of these time series is most useful for modeling COVID-19.

## COVID-19 Cases and Deaths - ECDC

We integrate data directly from the European Centre for Disease Prevention and Control (ECDC), a Stockholm-based organization with the aim of strengthening Europe's infectious disease preparedness. We do this because it is considered the most trustworthy international sources of accurate numbers, according to OurWorldInData.org. While traditionally a global pandemic would be tracked and coordinated via the World Health Organzation (WHO), the WHO has struggled both tactically and strategically to contribute meaningfully to the fight against COVID-19. Tactically, OurWorldInData found and reported to the organization that their data files contain many errors and are contradicted by the WHO's own Situation Reports. Additionally, a reporting cut-off time change between Reports 57 and 58 means there is data overlap/duplication that is impossible to eliminate. Strategically, the accuracy of the numbers that WHO is reporting have been challenged.

Source: https://www.ecdc.europa.eu/en/covid-19-pandemic


```python
# Download the latest CSV directly from ECDC website
ecdc_country = pd.read_csv("https://opendata.ecdc.europa.eu/covid19/casedistribution/csv", na_values = "", encoding="utf-8-sig")
ecdc_country['dateRep'] = pd.to_datetime(ecdc_country['dateRep'], format='%d/%m/%Y')
ecdc_country = ecdc_country.sort_values(by=['countriesAndTerritories', 'year', 'month', 'day'], ascending=True)\
                           .rename(columns={
                                    'dateRep':'date',
                                    'countriesAndTerritories': 'country',
                                    'geoId': 'geoID',
                                    'countryterritoryCode': 'country_code',
                                    'popData2018': 'population_2018',
                                    'continentExp': 'continent',
                                    'cases': 'new_cases',
                                    'deaths': 'new_deaths'})
display(ecdc_country.shape)
ecdc_country
```


    (16114, 11)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>day</th>
      <th>month</th>
      <th>year</th>
      <th>new_cases</th>
      <th>new_deaths</th>
      <th>country</th>
      <th>geoID</th>
      <th>country_code</th>
      <th>population_2018</th>
      <th>continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>121</td>
      <td>2019-12-31</td>
      <td>31</td>
      <td>12</td>
      <td>2019</td>
      <td>0</td>
      <td>0</td>
      <td>Afghanistan</td>
      <td>AF</td>
      <td>AFG</td>
      <td>37172386.0</td>
      <td>Asia</td>
    </tr>
    <tr>
      <td>120</td>
      <td>2020-01-01</td>
      <td>1</td>
      <td>1</td>
      <td>2020</td>
      <td>0</td>
      <td>0</td>
      <td>Afghanistan</td>
      <td>AF</td>
      <td>AFG</td>
      <td>37172386.0</td>
      <td>Asia</td>
    </tr>
    <tr>
      <td>119</td>
      <td>2020-01-02</td>
      <td>2</td>
      <td>1</td>
      <td>2020</td>
      <td>0</td>
      <td>0</td>
      <td>Afghanistan</td>
      <td>AF</td>
      <td>AFG</td>
      <td>37172386.0</td>
      <td>Asia</td>
    </tr>
    <tr>
      <td>118</td>
      <td>2020-01-03</td>
      <td>3</td>
      <td>1</td>
      <td>2020</td>
      <td>0</td>
      <td>0</td>
      <td>Afghanistan</td>
      <td>AF</td>
      <td>AFG</td>
      <td>37172386.0</td>
      <td>Asia</td>
    </tr>
    <tr>
      <td>117</td>
      <td>2020-01-04</td>
      <td>4</td>
      <td>1</td>
      <td>2020</td>
      <td>0</td>
      <td>0</td>
      <td>Afghanistan</td>
      <td>AF</td>
      <td>AFG</td>
      <td>37172386.0</td>
      <td>Asia</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>16067</td>
      <td>2020-05-06</td>
      <td>6</td>
      <td>5</td>
      <td>2020</td>
      <td>0</td>
      <td>0</td>
      <td>Zimbabwe</td>
      <td>ZW</td>
      <td>ZWE</td>
      <td>14439018.0</td>
      <td>Africa</td>
    </tr>
    <tr>
      <td>16066</td>
      <td>2020-05-07</td>
      <td>7</td>
      <td>5</td>
      <td>2020</td>
      <td>0</td>
      <td>0</td>
      <td>Zimbabwe</td>
      <td>ZW</td>
      <td>ZWE</td>
      <td>14439018.0</td>
      <td>Africa</td>
    </tr>
    <tr>
      <td>16065</td>
      <td>2020-05-08</td>
      <td>8</td>
      <td>5</td>
      <td>2020</td>
      <td>0</td>
      <td>0</td>
      <td>Zimbabwe</td>
      <td>ZW</td>
      <td>ZWE</td>
      <td>14439018.0</td>
      <td>Africa</td>
    </tr>
    <tr>
      <td>16064</td>
      <td>2020-05-09</td>
      <td>9</td>
      <td>5</td>
      <td>2020</td>
      <td>1</td>
      <td>0</td>
      <td>Zimbabwe</td>
      <td>ZW</td>
      <td>ZWE</td>
      <td>14439018.0</td>
      <td>Africa</td>
    </tr>
    <tr>
      <td>16063</td>
      <td>2020-05-10</td>
      <td>10</td>
      <td>5</td>
      <td>2020</td>
      <td>1</td>
      <td>0</td>
      <td>Zimbabwe</td>
      <td>ZW</td>
      <td>ZWE</td>
      <td>14439018.0</td>
      <td>Africa</td>
    </tr>
  </tbody>
</table>
<p>16114 rows × 11 columns</p>
</div>




```python
# Define function to standardize target country names
def standardize_country_names(dataframe, axis=1):
    country_names_dict = {
        'United States': 'United_States',
        'United_States_of_America': 'United_States',
        'South Korea': 'South_Korea',
        'Republic of Korea': 'South_Korea',
        'United Kingdom': 'United_Kingdom',
        'UK': 'United_Kingdom'
    }

    # Rename columns
    if axis==1:
        dataframe = dataframe.rename(columns=country_names_dict)
    # Rename row values
    elif axis==0:
        dict_iter = iter(country_names_dict.items())
        for i in range(len(country_names_dict)):
            pair = next(dict_iter)
            dataframe['country'] = dataframe['country'].replace(pair[0], pair[1])

    return dataframe
```


```python
# Rename select countries for legibility
ecdc_country = standardize_country_names(ecdc_country, axis=0)
```


```python
# Add columns to translate from new daily cases to cumulative cases
ecdc_country['cases'] = ecdc_country.groupby('country')['new_cases'].transform(pd.Series.cumsum)
ecdc_country['deaths'] = ecdc_country.groupby('country')['new_deaths'].transform(pd.Series.cumsum)
# ecdc_country
```

## Enrich ECDC Data

#### Standardize cases per 100,000 citizens


```python
# Cumulative cases per 100k inhabitants
ecdc_country['cases_per100k'] = ecdc_country['cases']*1e5/ecdc_country['population_2018']
ecdc_country['deaths_per100k'] = ecdc_country['deaths']*1e5/ecdc_country['population_2018']

ecdc_country['new_cases_per100k'] = ecdc_country['new_cases']*1e5/ecdc_country['population_2018']
ecdc_country['new_deaths_per100k'] = ecdc_country['new_deaths']*1e5/ecdc_country['population_2018']
```

#### Smooth data with rolling mean of daily new cases
Daily new case counts are highly erratic and we surmise that these developments follow a smoother true trend. To help the model learn the underlying problem rather than trying to memorize jagged lines, we smooth the response variable.

This was inspired by a precedent in Prof. Mauricio Santillana's published work.


```python
# Window == number of days in rolling window over which to calculate mean
window = 5
ecdc_country['new_cases_smooth'] = ecdc_country.groupby('country')['new_cases'].transform(lambda x: x.rolling(window).mean()).fillna(0)
ecdc_country['new_cases_per100k_smooth'] = ecdc_country.groupby('country')['new_cases_per100k'].transform(lambda x: x.rolling(window).mean()).fillna(0)
```

#### Calculate Day Zero

To standardize time series for various countries, we calculate Day Zero of COVID-19 infection for each day since a country's first case.


```python
# Calculate Day Zero for each country
day_zero = ecdc_country[ecdc_country.cases > 0].groupby('country')[['date']].first()
day_zero.columns = ['day_zero']
day_zero.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>day_zero</th>
    </tr>
    <tr>
      <th>country</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Afghanistan</td>
      <td>2020-02-25</td>
    </tr>
    <tr>
      <td>Albania</td>
      <td>2020-03-09</td>
    </tr>
    <tr>
      <td>Algeria</td>
      <td>2020-02-26</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Calculate Days Since for each entry for each country
ecdc_country = ecdc_country.join(day_zero, how='left', on='country')

ecdc_country['days_since_zero'] = (ecdc_country['date'] - ecdc_country['day_zero']).dt.days

ecdc_country[['country', 'days_since_zero']].tail(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>days_since_zero</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>16065</td>
      <td>Zimbabwe</td>
      <td>48</td>
    </tr>
    <tr>
      <td>16064</td>
      <td>Zimbabwe</td>
      <td>49</td>
    </tr>
    <tr>
      <td>16063</td>
      <td>Zimbabwe</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>



#### Preview final ECDC dataframe


```python
ecdc_country.tail(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>day</th>
      <th>month</th>
      <th>year</th>
      <th>new_cases</th>
      <th>new_deaths</th>
      <th>country</th>
      <th>geoID</th>
      <th>country_code</th>
      <th>population_2018</th>
      <th>...</th>
      <th>cases</th>
      <th>deaths</th>
      <th>cases_per100k</th>
      <th>deaths_per100k</th>
      <th>new_cases_per100k</th>
      <th>new_deaths_per100k</th>
      <th>new_cases_smooth</th>
      <th>new_cases_per100k_smooth</th>
      <th>day_zero</th>
      <th>days_since_zero</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>16065</td>
      <td>2020-05-08</td>
      <td>8</td>
      <td>5</td>
      <td>2020</td>
      <td>0</td>
      <td>0</td>
      <td>Zimbabwe</td>
      <td>ZW</td>
      <td>ZWE</td>
      <td>14439018.0</td>
      <td>...</td>
      <td>34</td>
      <td>4</td>
      <td>0.235473</td>
      <td>0.027703</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.428613e-18</td>
      <td>2020-03-21</td>
      <td>48</td>
    </tr>
    <tr>
      <td>16064</td>
      <td>2020-05-09</td>
      <td>9</td>
      <td>5</td>
      <td>2020</td>
      <td>1</td>
      <td>0</td>
      <td>Zimbabwe</td>
      <td>ZW</td>
      <td>ZWE</td>
      <td>14439018.0</td>
      <td>...</td>
      <td>35</td>
      <td>4</td>
      <td>0.242399</td>
      <td>0.027703</td>
      <td>0.006926</td>
      <td>0.0</td>
      <td>0.2</td>
      <td>1.385136e-03</td>
      <td>2020-03-21</td>
      <td>49</td>
    </tr>
    <tr>
      <td>16063</td>
      <td>2020-05-10</td>
      <td>10</td>
      <td>5</td>
      <td>2020</td>
      <td>1</td>
      <td>0</td>
      <td>Zimbabwe</td>
      <td>ZW</td>
      <td>ZWE</td>
      <td>14439018.0</td>
      <td>...</td>
      <td>36</td>
      <td>4</td>
      <td>0.249324</td>
      <td>0.027703</td>
      <td>0.006926</td>
      <td>0.0</td>
      <td>0.4</td>
      <td>2.770271e-03</td>
      <td>2020-03-21</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 21 columns</p>
</div>



### Define target countries and global variables


```python
# Make list of countries we're interested in analyzing
target_countries = [
    'France', 'Italy', 'Germany', 'Austria', 'United_Kingdom',
    'Spain', 'Portugal', 'Turkey', 'Norway', 'Sweden',
    'Belgium', 'Netherlands', 'Switzerland',
    'Singapore', 'United_States', 'Canada'
]
print(target_countries)
```

    ['France', 'Italy', 'Germany', 'Austria', 'United_Kingdom', 'Spain', 'Portugal', 'Turkey', 'Norway', 'Sweden', 'Belgium', 'Netherlands', 'Switzerland', 'Singapore', 'United_States', 'Canada']



```python
# Gather unique countries
unique_countries = ecdc_country.country.unique()
print(len(unique_countries), "unique countries")
```

    209 unique countries



```python
# Calculate greatest length of non-null time series i.e. longest period of COVID-19 presence
max_days = ecdc_country.days_since_zero.max()
print(max_days, "days in the longest impacted country")
```

    131 days in the longest impacted country


## Alternate Time Series Integration

### Integrate testing data - Source: OurWorldInData


```python
# Read CSV
testing_raw = pd.read_csv('covid-testing-all-observations.csv', na_values="")\
                .rename(columns={
                        'Cumulative total': 'tests',
                        'Daily change in cumulative total': 'new_tests',
                        'Date': 'date'})
testing_raw.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Entity</th>
      <th>ISO code</th>
      <th>date</th>
      <th>Source URL</th>
      <th>Source label</th>
      <th>Notes</th>
      <th>tests</th>
      <th>new_tests</th>
      <th>Cumulative total per thousand</th>
      <th>Daily change in cumulative total per thousand</th>
      <th>3-day rolling mean daily change</th>
      <th>3-day rolling mean daily change per thousand</th>
      <th>7-day rolling mean daily change</th>
      <th>7-day rolling mean daily change per thousand</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Argentina - tests performed</td>
      <td>ARG</td>
      <td>2020-04-08</td>
      <td>https://www.argentina.gob.ar/sites/default/fil...</td>
      <td>Government of Argentina</td>
      <td>NaN</td>
      <td>13330</td>
      <td>NaN</td>
      <td>0.295</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Argentina - tests performed</td>
      <td>ARG</td>
      <td>2020-04-09</td>
      <td>https://www.argentina.gob.ar/sites/default/fil...</td>
      <td>Government of Argentina</td>
      <td>NaN</td>
      <td>14850</td>
      <td>1520.0</td>
      <td>0.329</td>
      <td>0.034</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Argentina - tests performed</td>
      <td>ARG</td>
      <td>2020-04-10</td>
      <td>https://www.argentina.gob.ar/sites/default/fil...</td>
      <td>Government of Argentina</td>
      <td>NaN</td>
      <td>16379</td>
      <td>1529.0</td>
      <td>0.362</td>
      <td>0.034</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Define country column
country_column = testing_raw['Entity'].str.split(" - ", n=1, expand=True)
testing_raw['country'] = country_column[0]

# Rename select countries for legibility
testing_raw = standardize_country_names(testing_raw, axis=0)

# Calculate unique
unique_countries_testing = testing_raw['country'].unique()
print(len(unique_countries_testing), "unique countries with data on testing rates")
```

    84 unique countries with data on testing rates



```python
# The data provided records testing numbers from the CDC and a volunteer effort called the COVID Tracking Project.
# The CDC reports only COVID tests performed in public labs, while CTP reports all tests public or private
# by aggregating reported numbers from individual states. Because it is the superset, we elect to use that.

# Filter out duplicative CDC numbers
pre = len(testing_raw)
testing_raw = testing_raw[testing_raw['Source label'] != 'United States CDC']
post = len(testing_raw)
print("Remove CDC numbers:", pre, "rows => ", post)

# When two numbers for a single country are reported, select tests performed over people tested
# Tests performed is typically reported earlier and has more days of data
testing_raw = testing_raw[testing_raw['Entity'].isin([
    'India - people tested',
    'Italy - people tested',
    'Japan - people tested',
    'Singapore - people tested',
    'United Kingdom - people tested'
]) == False]

print("Remove repetitive numbers on people tested: rows => ", len(testing_raw))
```

    Remove CDC numbers: 4465 rows =>  4361
    Remove repetitive numbers on people tested: rows =>  4132


### Integrate Mobility Data - Source: Apple


```python
# Read CSV
apple_raw = pd.read_csv('applemobilitytrends.csv', na_values="")\
                .rename(columns={
                        'region': 'country'})
apple_raw.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>geo_type</th>
      <th>country</th>
      <th>transportation_type</th>
      <th>alternative_name</th>
      <th>2020-01-13</th>
      <th>2020-01-14</th>
      <th>2020-01-15</th>
      <th>2020-01-16</th>
      <th>2020-01-17</th>
      <th>2020-01-18</th>
      <th>...</th>
      <th>2020-04-29</th>
      <th>2020-04-30</th>
      <th>2020-05-01</th>
      <th>2020-05-02</th>
      <th>2020-05-03</th>
      <th>2020-05-04</th>
      <th>2020-05-05</th>
      <th>2020-05-06</th>
      <th>2020-05-07</th>
      <th>2020-05-08</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>country/region</td>
      <td>Albania</td>
      <td>driving</td>
      <td>NaN</td>
      <td>100.0</td>
      <td>95.30</td>
      <td>101.43</td>
      <td>97.20</td>
      <td>103.55</td>
      <td>112.67</td>
      <td>...</td>
      <td>35.90</td>
      <td>38.09</td>
      <td>37.23</td>
      <td>29.36</td>
      <td>36.00</td>
      <td>43.69</td>
      <td>42.61</td>
      <td>43.11</td>
      <td>46.13</td>
      <td>45.78</td>
    </tr>
    <tr>
      <td>1</td>
      <td>country/region</td>
      <td>Albania</td>
      <td>walking</td>
      <td>NaN</td>
      <td>100.0</td>
      <td>100.68</td>
      <td>98.93</td>
      <td>98.46</td>
      <td>100.85</td>
      <td>100.13</td>
      <td>...</td>
      <td>41.49</td>
      <td>38.25</td>
      <td>38.68</td>
      <td>32.28</td>
      <td>43.41</td>
      <td>49.59</td>
      <td>46.44</td>
      <td>52.84</td>
      <td>52.37</td>
      <td>48.10</td>
    </tr>
    <tr>
      <td>2</td>
      <td>country/region</td>
      <td>Argentina</td>
      <td>driving</td>
      <td>NaN</td>
      <td>100.0</td>
      <td>97.07</td>
      <td>102.45</td>
      <td>111.21</td>
      <td>118.45</td>
      <td>124.01</td>
      <td>...</td>
      <td>32.22</td>
      <td>34.45</td>
      <td>22.69</td>
      <td>28.23</td>
      <td>16.44</td>
      <td>32.01</td>
      <td>33.63</td>
      <td>35.13</td>
      <td>35.56</td>
      <td>40.25</td>
    </tr>
    <tr>
      <td>3</td>
      <td>country/region</td>
      <td>Argentina</td>
      <td>walking</td>
      <td>NaN</td>
      <td>100.0</td>
      <td>95.11</td>
      <td>101.37</td>
      <td>112.67</td>
      <td>116.72</td>
      <td>114.14</td>
      <td>...</td>
      <td>22.78</td>
      <td>24.80</td>
      <td>16.62</td>
      <td>20.61</td>
      <td>12.44</td>
      <td>21.35</td>
      <td>22.63</td>
      <td>23.84</td>
      <td>23.84</td>
      <td>30.63</td>
    </tr>
    <tr>
      <td>4</td>
      <td>country/region</td>
      <td>Australia</td>
      <td>driving</td>
      <td>NaN</td>
      <td>100.0</td>
      <td>102.98</td>
      <td>104.21</td>
      <td>108.63</td>
      <td>109.08</td>
      <td>89.00</td>
      <td>...</td>
      <td>59.46</td>
      <td>63.12</td>
      <td>58.40</td>
      <td>48.67</td>
      <td>58.18</td>
      <td>62.51</td>
      <td>64.04</td>
      <td>66.19</td>
      <td>71.34</td>
      <td>67.64</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 121 columns</p>
</div>




```python
# Rename select countries for legibility
apple_raw = standardize_country_names(apple_raw, axis=0)
```


```python
# Types of transit
apple_raw.transportation_type.unique()
```




    array(['driving', 'walking', 'transit'], dtype=object)



### Integrate Mobility Data - Source: Google


```python
# Read CSV
google_raw = pd.read_csv('Global_Mobility_Report.csv', na_values="")\
                .rename(columns={
                        'country_region': 'country'})
google_raw.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country_region_code</th>
      <th>country</th>
      <th>sub_region_1</th>
      <th>sub_region_2</th>
      <th>date</th>
      <th>retail_and_recreation_percent_change_from_baseline</th>
      <th>grocery_and_pharmacy_percent_change_from_baseline</th>
      <th>parks_percent_change_from_baseline</th>
      <th>transit_stations_percent_change_from_baseline</th>
      <th>workplaces_percent_change_from_baseline</th>
      <th>residential_percent_change_from_baseline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>AE</td>
      <td>United Arab Emirates</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2020-02-15</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>AE</td>
      <td>United Arab Emirates</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2020-02-16</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>AE</td>
      <td>United Arab Emirates</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2020-02-17</td>
      <td>-1.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>AE</td>
      <td>United Arab Emirates</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2020-02-18</td>
      <td>-2.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>AE</td>
      <td>United Arab Emirates</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2020-02-19</td>
      <td>-2.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>-1.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Rename select countries for legibility
google_raw = standardize_country_names(google_raw, axis=0)
```


```python
# Average mobility changes across types
google_raw['overall_change_from_baseline'] = round(np.mean(google_raw.iloc[:,5:], axis=1), 3)
```

## Helper Functions to improve Time Series Forecasting

#### MinMax Scaling


```python
# Define function to scale a dataframe
def scale(dataframe):
    scaler = MinMaxScaler()
    dataframe_scaled = scaler.fit_transform(dataframe)
    dataframe_scaled = pd.DataFrame(dataframe_scaled, columns=dataframe.columns, index=dataframe.index)

    return dataframe_scaled
```

#### X/Y Split for Supervised Learning
The below function takes in a `time_series_df` dataframe with the original data and returns a tensor with overlapping input and output sequences of some input length and output length for prediction. It takes the parameter `days_train` which specifies the temporal dimension of input sequences, i.e. how many days of history are used for each round of prediction. Similarly, `days_pred` specifies the length of the response sequence, i.e. how far into the future the model is to predict.

Which column to use as the response can be passed as a parameter, along with whether this column should be included in the `X` tensor.

For instance, `days_train = 14` and `days_pred = 7` means the model is to use 14 days of history to predict 7 days out. If the response is included, this means the response variable for days 1 through 14 is included, and we are to predict the response for days 15 through 21. The returned sequences step forward one day at a time, so the first input sequence is days 1-14, the second 2-15, etc.


```python
# Define function to create X/Y split of time series
# Permits option of training on response or holding out response
def structure_supervised(time_series_df, response:str, days_train, days_pred, holdout=False, print_shape=False):

    # Initialize x, y arrays as zeroes
    # Decide if you want to hold out response variable
    if holdout is True:
        x = np.zeros((0, days_train, time_series_df.shape[-1]-1)) # -1 because we remove the response from X
    elif holdout is False:
        x = np.zeros((0, days_train, time_series_df.shape[-1]))
    y = np.zeros((0, days_pred))

    # Check response variable entered correctly
    if type(response) != str:
        print('Please enter your target variable as a string')

    # For every row in the dataframe
    for i in range(len(time_series_df)):

        # Define the windows of training and prediction
        idx_in = i + days_train
        idx_out = idx_in + days_pred

        # If window of prediction tries to go past end of data, stop
        if idx_out > len(time_series_df):
            break

        # Choose to holdout response
        if holdout is True:
            # Create windowed training sequence for all non-response columns
            seq_in = np.array(time_series_df.iloc[i:idx_in, time_series_df.columns != response])

            # Create windowed prediction sequence for response column
            response_idx = time_series_df.columns.get_loc(response)
            seq_out = np.array(time_series_df.iloc[idx_in:idx_out, response_idx])
        elif holdout is False:
            try:    
                seq_in = np.array(time_series_df.iloc[i:idx_in, :])
                seq_out = np.array(time_series_df.iloc[idx_in:idx_out, -1])
            except AttributeError:
                seq_in = time_series_df[i:idx_in, :]
                seq_out = time_series_df[idx_in:idx_out, -1]

        # Add sequences to respective array, shaped for supervised learning
        x = np.concatenate((x, seq_in.reshape(((1,)+seq_in.shape))), axis = 0)
        y = np.concatenate((y, seq_out.reshape(((1,)+seq_out.shape))), axis = 0)

    # Decide if you want to print the output shape for clarity
    if print_shape == True:
        print("x_train shape: ", x.shape, "y_train shape: ", y.shape)
    return np.array(x), np.array(y)
```

---------

---------

# Generalizable Model for COVID-19 Population-Level Growth Rates

Our first challenge is to design a model that can use all available time series to best approximate generalizable coronavirus growth curves and provide predictions a sufficient number of days ahead for policymakers for any country regardless of the shape of its curve. We explore two architectures (Stacked LSTM and CNN) before identifying a combination design as the most effective at predicting future cases.

We aim to train a model to make predictions `m` days out based on the preceding `n` days of history, mapping input features to predictions on that horizon for the response, independent of the specific country. We have a lot of countries, but each `n`-length input sequence to `m`-length output sequence must contain consecutive days within the same country. Hence, countries with a longer (recorded) history of COVID-19 will be more represented in the dataset.

LSTM is specifically engineered for sequential data, and is thus a natural candidate for time series problems such as this. Assuming overfitting is addressed, e.g. via regularisation, stacking several layers of LSTM will make the model even better, though it also tends to make it very slow to train. We could alternatively use GRU layers, which sacrifice some predictive power for much faster training. However, we have on the order of a thousand training observations, so we can afford both to use LSTM and to stack consecutive layers for greater expressiveness, as training times will still be reasonable.

An alternative is using CNN for time series predictions. CNNs are translation invariant, meaning they do not capture sequential development the same way, but rather each filter learns to look for specific development patterns across the input sequence. It may sound unintuitive, but CNNs have shown impressive performance in several time series domains, from financial time series to ride-hailing demand prediction. Since we are trying to build a model to learn general infection rate curve development patterns, such an essentially curve-fitting model might prove effective.

That said, our primary motivation for incorporating CNN is as a pre-processing step for LSTM. Several models in academia and industry have found that passing input sequences through a CNN before passing them on to an LSTM improves overall predictive power. There are two ways to use CNN for pre-processing, both using 1D convolutions: the early CNN layers use 1D kernels of size greater than 1 to peruse the raw data for general translation invariant patterns and pass what it finds on to the LSTM; alternatively the early CNN layers use kernels to change the dimensionality of the input data, which in e.g. InceptionNet is used to reduce dimensionality, but in our case essentially performs feature engineering to increase the dimensionality such that LSTM has more varied representations of the input data to analyse. In both cases the CNN layers pass what they find on to LSTM, which maps the sequence of activations to the response sequences.


```python
# Limit to predictors of interest and observations after COVID-19 found in country
growth_curve = ecdc_country.loc[ecdc_country.days_since_zero >= 0,
                                [
                                    'country', 'date', 'new_cases', 'new_deaths',
                                    'days_since_zero', 'cases', 'deaths',
                                    'population_2018', 'day_zero',
                                    'cases_per100k', 'new_cases_per100k',
                                    'new_deaths_per100k', 'new_cases_smooth',
                                    'new_cases_per100k_smooth'
                                ]]
growth_curve.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>date</th>
      <th>new_cases</th>
      <th>new_deaths</th>
      <th>days_since_zero</th>
      <th>cases</th>
      <th>deaths</th>
      <th>population_2018</th>
      <th>day_zero</th>
      <th>cases_per100k</th>
      <th>new_cases_per100k</th>
      <th>new_deaths_per100k</th>
      <th>new_cases_smooth</th>
      <th>new_cases_per100k_smooth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>65</td>
      <td>Afghanistan</td>
      <td>2020-02-25</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>37172386.0</td>
      <td>2020-02-25</td>
      <td>0.00269</td>
      <td>0.00269</td>
      <td>0.0</td>
      <td>0.2</td>
      <td>0.000538</td>
    </tr>
    <tr>
      <td>64</td>
      <td>Afghanistan</td>
      <td>2020-02-26</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>37172386.0</td>
      <td>2020-02-25</td>
      <td>0.00269</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.2</td>
      <td>0.000538</td>
    </tr>
    <tr>
      <td>63</td>
      <td>Afghanistan</td>
      <td>2020-02-27</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>37172386.0</td>
      <td>2020-02-25</td>
      <td>0.00269</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.2</td>
      <td>0.000538</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Subset to countries of interest
growth_curve = growth_curve.loc[growth_curve.country.isin(target_countries)]
```

#### Visualize individual country curves


```python
# Plot country growth curves to visualize shape
plt.figure(figsize=(20,10))
for c in growth_curve.country.unique():
    plt.plot(growth_curve.loc[growth_curve.country == c,
                             ['days_since_zero', 'cases_per100k']]
                             .set_index('days_since_zero'),
                             label=c)

plt.title('Cases per 100k Citizens', size=20)
plt.legend(fontsize=12, loc=2)
plt.show()
```


![png](images/output_48_0.png)



```python
# Plot country growth curves to visualize shape
plt.figure(figsize=(20,10))
for c in growth_curve.country.unique():
    plt.plot(growth_curve.loc[growth_curve.country == c,
                             ['days_since_zero', 'new_cases_per100k_smooth']]
                             .set_index('days_since_zero'),
                             label=c)

plt.title('New Cases per 100k Citizens (smooth)', size=20)
plt.legend(fontsize=12, loc=2)
plt.show()
```


![png](images/output_49_0.png)


#### Prepare additional dataframes for enrichment
I.e. combine the different datasets into one for training and prediction.


```python
### Create structured testing dataframe
testing_curve = testing_raw[['date', 'country', 'tests', 'new_tests']]

# Translate date column to datetime for joining
testing_curve.loc[:, 'date'] = pd.to_datetime(testing_raw['date'])

# Merge to Growth Curve All
growth_curve_all = growth_curve.merge(testing_curve, how='left', on=['date', 'country'])
assert len(growth_curve_all) == len(growth_curve)


# Impute zeroes for new tests
growth_curve_all.loc[:, 'new_tests'] = growth_curve_all.new_tests.fillna(0)

# Forward fill recently missing date for tests
print('Original missing tests:', growth_curve_all.tests.isna().sum())
# Within each country, forward-fill after first non-missing total tests
growth_curve_all.loc[:, 'tests'] = growth_curve_all.groupby('country')['tests'].transform(lambda x: x.ffill())
print('After forward-filling:', growth_curve_all.tests.isna().sum())
# Then impute 0 for any remaining missing tests, as that will be because testing was not yet performed
growth_curve_all.tests.fillna(0, inplace = True)
print('After imputing zeros:', growth_curve_all.tests.isna().sum())

# Derive new standardized columns
growth_curve_all['tests_per100k'] = growth_curve_all.tests / growth_curve_all.population_2018
growth_curve_all['new_tests_per100k'] = growth_curve_all.new_tests / growth_curve_all.population_2018

growth_curve_all.tail(7)
```

    Original missing tests: 788
    After forward-filling: 497
    After imputing zeros: 0





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>date</th>
      <th>new_cases</th>
      <th>new_deaths</th>
      <th>days_since_zero</th>
      <th>cases</th>
      <th>deaths</th>
      <th>population_2018</th>
      <th>day_zero</th>
      <th>cases_per100k</th>
      <th>new_cases_per100k</th>
      <th>new_deaths_per100k</th>
      <th>new_cases_smooth</th>
      <th>new_cases_per100k_smooth</th>
      <th>tests</th>
      <th>new_tests</th>
      <th>tests_per100k</th>
      <th>new_tests_per100k</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1451</td>
      <td>United_States</td>
      <td>2020-05-04</td>
      <td>24972</td>
      <td>1297</td>
      <td>104</td>
      <td>1158041</td>
      <td>67682</td>
      <td>327167434.0</td>
      <td>2020-01-21</td>
      <td>353.959740</td>
      <td>7.632789</td>
      <td>0.396433</td>
      <td>29091.6</td>
      <td>8.891961</td>
      <td>7285374.0</td>
      <td>232008.0</td>
      <td>0.022268</td>
      <td>0.000709</td>
    </tr>
    <tr>
      <td>1452</td>
      <td>United_States</td>
      <td>2020-05-05</td>
      <td>22593</td>
      <td>1252</td>
      <td>105</td>
      <td>1180634</td>
      <td>68934</td>
      <td>327167434.0</td>
      <td>2020-01-21</td>
      <td>360.865379</td>
      <td>6.905638</td>
      <td>0.382679</td>
      <td>28145.0</td>
      <td>8.602629</td>
      <td>7544328.0</td>
      <td>258954.0</td>
      <td>0.023060</td>
      <td>0.000792</td>
    </tr>
    <tr>
      <td>1453</td>
      <td>United_States</td>
      <td>2020-05-06</td>
      <td>23841</td>
      <td>2144</td>
      <td>106</td>
      <td>1204475</td>
      <td>71078</td>
      <td>327167434.0</td>
      <td>2020-01-21</td>
      <td>368.152473</td>
      <td>7.287094</td>
      <td>0.655322</td>
      <td>26929.8</td>
      <td>8.231198</td>
      <td>7786793.0</td>
      <td>242465.0</td>
      <td>0.023801</td>
      <td>0.000741</td>
    </tr>
    <tr>
      <td>1454</td>
      <td>United_States</td>
      <td>2020-05-07</td>
      <td>24128</td>
      <td>2353</td>
      <td>107</td>
      <td>1228603</td>
      <td>73431</td>
      <td>327167434.0</td>
      <td>2020-01-21</td>
      <td>375.527290</td>
      <td>7.374817</td>
      <td>0.719204</td>
      <td>24964.4</td>
      <td>7.630466</td>
      <td>8105513.0</td>
      <td>318720.0</td>
      <td>0.024775</td>
      <td>0.000974</td>
    </tr>
    <tr>
      <td>1455</td>
      <td>United_States</td>
      <td>2020-05-08</td>
      <td>28369</td>
      <td>2239</td>
      <td>108</td>
      <td>1256972</td>
      <td>75670</td>
      <td>327167434.0</td>
      <td>2020-01-21</td>
      <td>384.198386</td>
      <td>8.671095</td>
      <td>0.684359</td>
      <td>24780.6</td>
      <td>7.574287</td>
      <td>8408788.0</td>
      <td>303275.0</td>
      <td>0.025702</td>
      <td>0.000927</td>
    </tr>
    <tr>
      <td>1456</td>
      <td>United_States</td>
      <td>2020-05-09</td>
      <td>26957</td>
      <td>1510</td>
      <td>109</td>
      <td>1283929</td>
      <td>77180</td>
      <td>327167434.0</td>
      <td>2020-01-21</td>
      <td>392.437898</td>
      <td>8.239512</td>
      <td>0.461537</td>
      <td>25177.6</td>
      <td>7.695631</td>
      <td>8709630.0</td>
      <td>300842.0</td>
      <td>0.026621</td>
      <td>0.000920</td>
    </tr>
    <tr>
      <td>1457</td>
      <td>United_States</td>
      <td>2020-05-10</td>
      <td>25612</td>
      <td>1614</td>
      <td>110</td>
      <td>1309541</td>
      <td>78794</td>
      <td>327167434.0</td>
      <td>2020-01-21</td>
      <td>400.266305</td>
      <td>7.828408</td>
      <td>0.493325</td>
      <td>25781.4</td>
      <td>7.880185</td>
      <td>8709630.0</td>
      <td>0.0</td>
      <td>0.026621</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create necessary mobility dataframe from just Apple
apple_long = apple_raw.drop(['geo_type', 'alternative_name'], axis = 1)

# Filter to only target countries
apple_long = apple_long.loc[apple_long.country.isin(growth_curve_all.country.unique())]

# Transpose date columns to rows
apple_long = apple_long.melt(id_vars = ['country', 'transportation_type'], var_name = 'date')
apple_long.sort_values(['country', 'date', 'transportation_type'])
apple_long.loc[:, 'date'] = pd.to_datetime(apple_long['date'])
# display(apple_long.shape)
display(apple_long.head(3))

# Translate so each row is country and date
apple_long = apple_long.pivot_table(values = 'value', index = ['country', 'date'], columns = 'transportation_type')
# Split back out original columns
apple_long = apple_long.reset_index()
# display(apple_long.shape)
# display(apple_long.head(3))

# Remove transit since missing lots of data
apple_long.drop(['transit'], axis = 1, inplace = True)

# Rename columns
apple_long.columns = ['country', 'date', 'mobility_drive', 'mobility_walk']

# Drop duplicates
apple_long.drop_duplicates(subset = ['country', 'date'], inplace = True)

# Merge to Growth Curve All
growth_curve_all = growth_curve_all.merge(apple_long, how = 'left', on = ['date', 'country'])
display(growth_curve_all.shape)

print('Total missing values:', growth_curve_all.isna().sum().sum())
print('Columns missing values:', list(growth_curve_all.isna().sum().index[growth_curve_all.isna().sum() > 0]))

# Fill forward missing values
growth_curve_all.fillna(method = 'ffill', inplace = True)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>transportation_type</th>
      <th>date</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Austria</td>
      <td>driving</td>
      <td>2020-01-13</td>
      <td>100.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Austria</td>
      <td>walking</td>
      <td>2020-01-13</td>
      <td>100.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Belgium</td>
      <td>driving</td>
      <td>2020-01-13</td>
      <td>100.0</td>
    </tr>
  </tbody>
</table>
</div>



    (1458, 20)


    Total missing values: 62
    Columns missing values: ['mobility_drive', 'mobility_walk']



```python
# Display final dataframe
print(growth_curve_all.shape)
growth_curve_all.tail(7)
```

    (1458, 20)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>date</th>
      <th>new_cases</th>
      <th>new_deaths</th>
      <th>days_since_zero</th>
      <th>cases</th>
      <th>deaths</th>
      <th>population_2018</th>
      <th>day_zero</th>
      <th>cases_per100k</th>
      <th>new_cases_per100k</th>
      <th>new_deaths_per100k</th>
      <th>new_cases_smooth</th>
      <th>new_cases_per100k_smooth</th>
      <th>tests</th>
      <th>new_tests</th>
      <th>tests_per100k</th>
      <th>new_tests_per100k</th>
      <th>mobility_drive</th>
      <th>mobility_walk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1451</td>
      <td>United_States</td>
      <td>2020-05-04</td>
      <td>24972</td>
      <td>1297</td>
      <td>104</td>
      <td>1158041</td>
      <td>67682</td>
      <td>327167434.0</td>
      <td>2020-01-21</td>
      <td>353.959740</td>
      <td>7.632789</td>
      <td>0.396433</td>
      <td>29091.6</td>
      <td>8.891961</td>
      <td>7285374.0</td>
      <td>232008.0</td>
      <td>0.022268</td>
      <td>0.000709</td>
      <td>78.01</td>
      <td>63.27</td>
    </tr>
    <tr>
      <td>1452</td>
      <td>United_States</td>
      <td>2020-05-05</td>
      <td>22593</td>
      <td>1252</td>
      <td>105</td>
      <td>1180634</td>
      <td>68934</td>
      <td>327167434.0</td>
      <td>2020-01-21</td>
      <td>360.865379</td>
      <td>6.905638</td>
      <td>0.382679</td>
      <td>28145.0</td>
      <td>8.602629</td>
      <td>7544328.0</td>
      <td>258954.0</td>
      <td>0.023060</td>
      <td>0.000792</td>
      <td>80.55</td>
      <td>64.79</td>
    </tr>
    <tr>
      <td>1453</td>
      <td>United_States</td>
      <td>2020-05-06</td>
      <td>23841</td>
      <td>2144</td>
      <td>106</td>
      <td>1204475</td>
      <td>71078</td>
      <td>327167434.0</td>
      <td>2020-01-21</td>
      <td>368.152473</td>
      <td>7.287094</td>
      <td>0.655322</td>
      <td>26929.8</td>
      <td>8.231198</td>
      <td>7786793.0</td>
      <td>242465.0</td>
      <td>0.023801</td>
      <td>0.000741</td>
      <td>81.47</td>
      <td>64.32</td>
    </tr>
    <tr>
      <td>1454</td>
      <td>United_States</td>
      <td>2020-05-07</td>
      <td>24128</td>
      <td>2353</td>
      <td>107</td>
      <td>1228603</td>
      <td>73431</td>
      <td>327167434.0</td>
      <td>2020-01-21</td>
      <td>375.527290</td>
      <td>7.374817</td>
      <td>0.719204</td>
      <td>24964.4</td>
      <td>7.630466</td>
      <td>8105513.0</td>
      <td>318720.0</td>
      <td>0.024775</td>
      <td>0.000974</td>
      <td>85.85</td>
      <td>68.90</td>
    </tr>
    <tr>
      <td>1455</td>
      <td>United_States</td>
      <td>2020-05-08</td>
      <td>28369</td>
      <td>2239</td>
      <td>108</td>
      <td>1256972</td>
      <td>75670</td>
      <td>327167434.0</td>
      <td>2020-01-21</td>
      <td>384.198386</td>
      <td>8.671095</td>
      <td>0.684359</td>
      <td>24780.6</td>
      <td>7.574287</td>
      <td>8408788.0</td>
      <td>303275.0</td>
      <td>0.025702</td>
      <td>0.000927</td>
      <td>95.56</td>
      <td>73.57</td>
    </tr>
    <tr>
      <td>1456</td>
      <td>United_States</td>
      <td>2020-05-09</td>
      <td>26957</td>
      <td>1510</td>
      <td>109</td>
      <td>1283929</td>
      <td>77180</td>
      <td>327167434.0</td>
      <td>2020-01-21</td>
      <td>392.437898</td>
      <td>8.239512</td>
      <td>0.461537</td>
      <td>25177.6</td>
      <td>7.695631</td>
      <td>8709630.0</td>
      <td>300842.0</td>
      <td>0.026621</td>
      <td>0.000920</td>
      <td>95.56</td>
      <td>73.57</td>
    </tr>
    <tr>
      <td>1457</td>
      <td>United_States</td>
      <td>2020-05-10</td>
      <td>25612</td>
      <td>1614</td>
      <td>110</td>
      <td>1309541</td>
      <td>78794</td>
      <td>327167434.0</td>
      <td>2020-01-21</td>
      <td>400.266305</td>
      <td>7.828408</td>
      <td>0.493325</td>
      <td>25781.4</td>
      <td>7.880185</td>
      <td>8709630.0</td>
      <td>0.0</td>
      <td>0.026621</td>
      <td>0.000000</td>
      <td>95.56</td>
      <td>73.57</td>
    </tr>
  </tbody>
</table>
</div>



#### Prepare Growth Curve dataframe for supervised learning


```python
# Filter to identified best predictors
predictors = ['days_since_zero', 'new_tests_per100k', 'tests_per100k',
              'new_deaths_per100k', 'cases_per100k', 'mobility_walk',
              'mobility_drive', 'new_cases_per100k_smooth']
features = len(predictors)
print(f"Keeping {features} features")
```

    Keeping 8 features


#### Structure as sequences


```python
# Define parameters for predictions length
days_train = 14
days_pred = 7
```


```python
# Available variables
for i, var in enumerate(predictors):
    print(i, var)
```

    0 days_since_zero
    1 new_tests_per100k
    2 tests_per100k
    3 new_deaths_per100k
    4 cases_per100k
    5 mobility_walk
    6 mobility_drive
    7 new_cases_per100k_smooth



```python
# Initalize empty arrays
train_x = np.zeros((0, days_train, features))
train_y = np.zeros((0, days_pred))

# Define response
response = predictors[7]
response_scale = max(response)
print(response)

# Scale data for improved accuracy
growth_curve_scaled = scale(growth_curve_all[predictors])

# For each country, create a supervised version of the data
for country in growth_curve_all.country.unique():
    print(country)
    # Filter to a target country
    gc_country = growth_curve_scaled[growth_curve_all.country == country]

    # Create x/y split
    gc_x, gc_y = structure_supervised(gc_country, response, days_train, days_pred)

    # Add to overall x/y_train
    train_x = np.concatenate((train_x, gc_x), axis = 0)
    train_y = np.concatenate((train_y, gc_y), axis = 0)

print(train_x.shape, train_y.shape)
```

    new_cases_per100k_smooth
    Austria
    Belgium
    Canada
    France
    Germany
    Italy
    Netherlands
    Norway
    Portugal
    Singapore
    Spain
    Sweden
    Switzerland
    Turkey
    United_Kingdom
    United_States
    (1138, 14, 8) (1138, 7)


#### Define plotting helper functions


```python
# Define function to plot all countries and their predictions
def plot_countries(model, response, ncols=4):

    # Define flexible plot shape
    country_count = len(growth_curve_all.country.unique())
    ncols = ncols
    nrows = int(np.floor(country_count/ncols))
    cut = country_count % nrows
    print(f"Plotting {country_count} countries on {ncols}x{nrows} grid with {cut} left out")

    fig, ax = plt.subplots(nrows, ncols, figsize=(20,5*nrows))

    # For each country, generate supervised structure and predictions
    row_idx = -1
    for i, c in enumerate(growth_curve_all.country.unique()):
        if i % ncols == 0:
            row_idx+=1
        if i+1 > (ncols*nrows):
            break

        gc_c = growth_curve_scaled[growth_curve_all.country == c]

        c_x, c_y = structure_supervised(gc_c, response, days_train, days_pred)

        # Calculate country-specific 2018 population to re-scale data
        c_pop = growth_curve_all[growth_curve_all.country == c]['population_2018'].iloc[0]
        max_count = max(growth_curve_all['new_cases_per100k_smooth'])

        c_obs = []
        for j in range(len(c_x)):
            c_obs.append(c_x[j, -1, -1])
        c_obs = np.array(c_obs) * max_count #* (c_pop/100000)

        c_pred = model.predict(c_x)
        c_pred = c_pred * max_count #* (c_pop/100000)

        ax[row_idx, i%ncols].plot(np.arange(len(c_obs)), np.array(c_obs), label = 'Observed')

        for j in range(len(c_pred)):
            ax[row_idx, i%ncols].plot(np.arange(j+1, j+1+c_pred.shape[-1]), c_pred[j, :], c = 'orange')

        ax[row_idx, i%ncols].set_title(c)
    plt.suptitle("Case Growth Predictions for Target Nations On Days Since First Case", fontsize=20, color='#237AB4')
    ax[0,0].set_ylabel("New cases per 100k citizens per day", fontsize=15, color='#237AB4')
    fig.subplots_adjust(top=0.93, hspace=0.25)
```


```python
# Define function to plot USA specific data
def plot_usa_last(model, shift = 0):
    pred = model.predict(usa_x)
    # Calculate numbers for re-scaling
    usa_pop = growth_curve_all[growth_curve_all.country == 'United_States']['population_2018'].iloc[0]
    max_count = max(growth_curve_all['new_cases_per100k_smooth'])
    pred = pred * max_count #* (usa_pop/100000)
    pred.shape

    cas_obs = []
    for i in range(len(usa_x)):
        cas_obs.append(usa_x[i, -1, -1])
    cas_obs = np.array(cas_obs) * max_count #* (usa_pop/100000)

    plt.figure(figsize = (20,10))
    plt.plot(np.arange(shift, len(cas_obs)), cas_obs[shift:], label = 'Observed')

    for i in range(shift, len(pred)):
        if i == len(pred)-1:
            plt.plot(np.arange(i, i+pred.shape[-1]), pred[i, :],
                     c = 'orange', ls = 'dashed', alpha = 1, label = str(pred.shape[-1])+'-day predictions')
        else:
            plt.plot(np.arange(i, i+pred.shape[-1]), pred[i, :],
                     c = 'orange', ls = 'dashed', alpha = 1)

    plt.legend()
    plt.title(f"USA Predictions", fontsize=20, color='#237AB4')
    plt.xlabel("Days Since First Case", fontsize=15, color='#237AB4')
    plt.ylabel("New cases per 100k citizens per day", fontsize=15, color='#237AB4')
    plt.show()
```


```python
# Plot model training history
def plot_training(history):
    plt.figure(figsize = (20,5))
    plt.plot(history.history['loss'], label = 'Training')
    plt.plot(history.history['val_loss'], label = 'Validation')
    plt.title('Training')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
```

### Define model structures for generalizable model fitting

#### Define prediction weights based on days in future
Since we are predicting `m` days into the future, our networks' last layer is always a Dense (i.e. fully-connected) layer with `m` nodes. By default these `m` nodes are weighted equally when calculating MSE across the `m` nodes and the observations in each batch.

We experimented extensively with weighting the `m` nodes differently to see whether that helped make the model more accurate for near-term or far-term prediction, but found the best results when we did not alter the weighting scheme. That said, if overall MSE is not top priority, the weighting scheme below can be adjusted to incentivise the model to prioritise accuracy in specific parts of the prediction sequence -- for instance to sacrifice far-future prediction accuracy for greatest accuracy 3 days out.


```python
#cw = dict(zip(np.arange(days_pred), np.linspace(3, 1, days_pred)))
#cw = dict(zip(np.arange(days_pred), np.logspace(0.5, 0, days_pred)))
cw = dict(zip(np.arange(days_pred), np.ones(days_pred)))
display(cw)
```


    {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0}



```python
# Create sample X/Y for United States
gc_usa = growth_curve_scaled[growth_curve_all.country == 'United_States']
print(response)
usa_x, usa_y = structure_supervised(gc_usa, response, days_train, days_pred)
```

    new_cases_per100k_smooth


#### Define Stacked LSTM for Population Level Growth


```python
# Define Stacked LSTM model
stacked_lstm = Sequential(name = 'LSTM_baseline')
stacked_lstm.add(LSTM(256, input_shape=(days_train, features),
             return_sequences=True, dropout = 0.0, recurrent_dropout = 0.4))
stacked_lstm.add(LSTM(128,
             return_sequences = False, dropout = 0.0, recurrent_dropout = 0.3))
stacked_lstm.add(Dense(days_pred))

stacked_lstm.compile(optimizer=Adam(learning_rate = 0.001/2), loss='mse', metric = ['rmse'])

stacked_lstm.summary()
```

    Model: "LSTM_baseline"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm (LSTM)                  (None, 14, 256)           271360    
    _________________________________________________________________
    lstm_1 (LSTM)                (None, 128)               197120    
    _________________________________________________________________
    dense (Dense)                (None, 7)                 903       
    =================================================================
    Total params: 469,383
    Trainable params: 469,383
    Non-trainable params: 0
    _________________________________________________________________



```python
# Fit Simple RNN to target country COVID-19 cases
history_stacked_lstm = stacked_lstm.fit(train_x, train_y, epochs=50, verbose=1, shuffle = True,
                      batch_size = 64, class_weight = cw, validation_split = 0.1)
```

    Train on 1024 samples, validate on 114 samples
    Epoch 1/50
    1024/1024 [==============================] - 4s 4ms/sample - loss: 0.0423 - val_loss: 0.0158
    Epoch 2/50
    1024/1024 [==============================] - 1s 576us/sample - loss: 0.0194 - val_loss: 0.0052
    Epoch 3/50
    1024/1024 [==============================] - 1s 568us/sample - loss: 0.0148 - val_loss: 0.0049
    Epoch 4/50
    1024/1024 [==============================] - 1s 555us/sample - loss: 0.0118 - val_loss: 0.0037
    Epoch 5/50
    1024/1024 [==============================] - 1s 566us/sample - loss: 0.0108 - val_loss: 0.0031
    Epoch 6/50
    1024/1024 [==============================] - 1s 557us/sample - loss: 0.0097 - val_loss: 0.0033
    Epoch 7/50
    1024/1024 [==============================] - 1s 563us/sample - loss: 0.0088 - val_loss: 0.0046
    Epoch 8/50
    1024/1024 [==============================] - 1s 566us/sample - loss: 0.0079 - val_loss: 0.0032
    Epoch 9/50
    1024/1024 [==============================] - 1s 569us/sample - loss: 0.0076 - val_loss: 0.0060
    Epoch 10/50
    1024/1024 [==============================] - 1s 565us/sample - loss: 0.0070 - val_loss: 0.0046
    Epoch 11/50
    1024/1024 [==============================] - 1s 559us/sample - loss: 0.0070 - val_loss: 0.0030
    Epoch 12/50
    1024/1024 [==============================] - 1s 570us/sample - loss: 0.0065 - val_loss: 0.0066
    Epoch 13/50
    1024/1024 [==============================] - 1s 568us/sample - loss: 0.0062 - val_loss: 0.0044
    Epoch 14/50
    1024/1024 [==============================] - 1s 570us/sample - loss: 0.0056 - val_loss: 0.0051
    Epoch 15/50
    1024/1024 [==============================] - 1s 571us/sample - loss: 0.0058 - val_loss: 0.0071
    Epoch 16/50
    1024/1024 [==============================] - 1s 566us/sample - loss: 0.0056 - val_loss: 0.0039
    Epoch 17/50
    1024/1024 [==============================] - 1s 588us/sample - loss: 0.0054 - val_loss: 0.0052
    Epoch 18/50
    1024/1024 [==============================] - 1s 643us/sample - loss: 0.0051 - val_loss: 0.0050
    Epoch 19/50
    1024/1024 [==============================] - 1s 631us/sample - loss: 0.0049 - val_loss: 0.0032
    Epoch 20/50
    1024/1024 [==============================] - 1s 583us/sample - loss: 0.0048 - val_loss: 0.0026
    Epoch 21/50
    1024/1024 [==============================] - 1s 558us/sample - loss: 0.0047 - val_loss: 0.0059
    Epoch 22/50
    1024/1024 [==============================] - 1s 565us/sample - loss: 0.0043 - val_loss: 0.0063
    Epoch 23/50
    1024/1024 [==============================] - 1s 568us/sample - loss: 0.0044 - val_loss: 0.0044
    Epoch 24/50
    1024/1024 [==============================] - 1s 569us/sample - loss: 0.0044 - val_loss: 0.0029
    Epoch 25/50
    1024/1024 [==============================] - 1s 566us/sample - loss: 0.0043 - val_loss: 0.0042
    Epoch 26/50
    1024/1024 [==============================] - 1s 559us/sample - loss: 0.0044 - val_loss: 0.0031
    Epoch 27/50
    1024/1024 [==============================] - 1s 565us/sample - loss: 0.0043 - val_loss: 0.0076
    Epoch 28/50
    1024/1024 [==============================] - 1s 548us/sample - loss: 0.0044 - val_loss: 0.0031
    Epoch 29/50
    1024/1024 [==============================] - 1s 554us/sample - loss: 0.0037 - val_loss: 0.0035
    Epoch 30/50
    1024/1024 [==============================] - 1s 555us/sample - loss: 0.0037 - val_loss: 0.0038
    Epoch 31/50
    1024/1024 [==============================] - 1s 566us/sample - loss: 0.0037 - val_loss: 0.0029
    Epoch 32/50
    1024/1024 [==============================] - 1s 560us/sample - loss: 0.0037 - val_loss: 0.0029
    Epoch 33/50
    1024/1024 [==============================] - 1s 566us/sample - loss: 0.0035 - val_loss: 0.0051
    Epoch 34/50
    1024/1024 [==============================] - 1s 562us/sample - loss: 0.0037 - val_loss: 0.0025
    Epoch 35/50
    1024/1024 [==============================] - 1s 558us/sample - loss: 0.0037 - val_loss: 0.0030
    Epoch 36/50
    1024/1024 [==============================] - 1s 562us/sample - loss: 0.0036 - val_loss: 0.0035
    Epoch 37/50
    1024/1024 [==============================] - 1s 566us/sample - loss: 0.0036 - val_loss: 0.0021
    Epoch 38/50
    1024/1024 [==============================] - 1s 565us/sample - loss: 0.0034 - val_loss: 0.0031
    Epoch 39/50
    1024/1024 [==============================] - 1s 555us/sample - loss: 0.0034 - val_loss: 0.0022
    Epoch 40/50
    1024/1024 [==============================] - 1s 559us/sample - loss: 0.0034 - val_loss: 0.0025
    Epoch 41/50
    1024/1024 [==============================] - 1s 553us/sample - loss: 0.0035 - val_loss: 0.0028
    Epoch 42/50
    1024/1024 [==============================] - 1s 548us/sample - loss: 0.0034 - val_loss: 0.0024
    Epoch 43/50
    1024/1024 [==============================] - 1s 612us/sample - loss: 0.0034 - val_loss: 0.0029
    Epoch 44/50
    1024/1024 [==============================] - 1s 624us/sample - loss: 0.0036 - val_loss: 0.0036
    Epoch 45/50
    1024/1024 [==============================] - 1s 614us/sample - loss: 0.0033 - val_loss: 0.0025
    Epoch 46/50
    1024/1024 [==============================] - 1s 558us/sample - loss: 0.0031 - val_loss: 0.0023
    Epoch 47/50
    1024/1024 [==============================] - 1s 554us/sample - loss: 0.0032 - val_loss: 0.0045
    Epoch 48/50
    1024/1024 [==============================] - 1s 558us/sample - loss: 0.0032 - val_loss: 0.0031
    Epoch 49/50
    1024/1024 [==============================] - 1s 558us/sample - loss: 0.0031 - val_loss: 0.0033
    Epoch 50/50
    1024/1024 [==============================] - 1s 557us/sample - loss: 0.0031 - val_loss: 0.0025



```python
# Plot training history
plot_training(history_stacked_lstm)
```


![png](images/output_71_0.png)



```python
# Plot 7-day model predictions for USA
plot_usa_last(stacked_lstm, 40)
```


![png](images/output_72_0.png)



```python
# Plot 7-day model predictions for target countries
plot_countries(stacked_lstm, response, ncols=3)
```

    Plotting 16 countries on 3x5 grid with 1 left out



![png](images/output_73_1.png)


#### Define 1-D CNN for Population Level Growth


```python
# Define simple 1D CNN model
cnn = Sequential(name = 'CNN_baseline')

cnn.add(Conv1D(filters = 64, kernel_size = 3, activation = 'relu',
               input_shape=(days_train, features)))
cnn.add(MaxPool1D(pool_size = 2))
cnn.add(Conv1D(filters = 128, kernel_size = 3, activation = 'relu'))
cnn.add(MaxPool1D(pool_size = 2))
cnn.add(Flatten())
cnn.add(Dense(256, activation = 'relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(128, activation = 'relu'))
cnn.add(Dropout(0.4))
cnn.add(Dense(64, activation = 'relu'))
cnn.add(Dropout(0.3))
cnn.add(Dense(days_pred))

cnn.compile(optimizer=Adam(learning_rate = 0.001/2), loss='mse', metric = ['rmse'])

cnn.summary()
```

    Model: "CNN_baseline"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv1d (Conv1D)              (None, 12, 64)            1600      
    _________________________________________________________________
    max_pooling1d (MaxPooling1D) (None, 6, 64)             0         
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 4, 128)            24704     
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 2, 128)            0         
    _________________________________________________________________
    flatten (Flatten)            (None, 256)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 256)               65792     
    _________________________________________________________________
    dropout (Dropout)            (None, 256)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 128)               32896     
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 128)               0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 64)                8256      
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 64)                0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 7)                 455       
    =================================================================
    Total params: 133,703
    Trainable params: 133,703
    Non-trainable params: 0
    _________________________________________________________________



```python
# Fit 1D CNN model
history_cnn = cnn.fit(train_x, train_y, epochs=100, verbose=1,
                      batch_size = 64, class_weight = cw, validation_split = 0.1)
```

    Train on 1024 samples, validate on 114 samples
    Epoch 1/100
    1024/1024 [==============================] - 1s 1ms/sample - loss: 0.0665 - val_loss: 0.0382
    Epoch 2/100
    1024/1024 [==============================] - 0s 81us/sample - loss: 0.0397 - val_loss: 0.0233
    Epoch 3/100
    1024/1024 [==============================] - 0s 82us/sample - loss: 0.0265 - val_loss: 0.0129
    Epoch 4/100
    1024/1024 [==============================] - 0s 82us/sample - loss: 0.0198 - val_loss: 0.0076
    Epoch 5/100
    1024/1024 [==============================] - 0s 81us/sample - loss: 0.0177 - val_loss: 0.0133
    Epoch 6/100
    1024/1024 [==============================] - 0s 82us/sample - loss: 0.0164 - val_loss: 0.0206
    Epoch 7/100
    1024/1024 [==============================] - 0s 81us/sample - loss: 0.0141 - val_loss: 0.0102
    Epoch 8/100
    1024/1024 [==============================] - 0s 80us/sample - loss: 0.0120 - val_loss: 0.0126
    Epoch 9/100
    1024/1024 [==============================] - 0s 81us/sample - loss: 0.0113 - val_loss: 0.0165
    Epoch 10/100
    1024/1024 [==============================] - 0s 81us/sample - loss: 0.0114 - val_loss: 0.0091
    Epoch 11/100
    1024/1024 [==============================] - 0s 83us/sample - loss: 0.0104 - val_loss: 0.0195
    Epoch 12/100
    1024/1024 [==============================] - 0s 81us/sample - loss: 0.0106 - val_loss: 0.0080
    Epoch 13/100
    1024/1024 [==============================] - 0s 85us/sample - loss: 0.0093 - val_loss: 0.0111
    Epoch 14/100
    1024/1024 [==============================] - 0s 84us/sample - loss: 0.0089 - val_loss: 0.0126
    Epoch 15/100
    1024/1024 [==============================] - 0s 91us/sample - loss: 0.0086 - val_loss: 0.0093
    Epoch 16/100
    1024/1024 [==============================] - 0s 91us/sample - loss: 0.0074 - val_loss: 0.0141
    Epoch 17/100
    1024/1024 [==============================] - 0s 88us/sample - loss: 0.0081 - val_loss: 0.0122
    Epoch 18/100
    1024/1024 [==============================] - 0s 94us/sample - loss: 0.0082 - val_loss: 0.0111
    Epoch 19/100
    1024/1024 [==============================] - 0s 90us/sample - loss: 0.0086 - val_loss: 0.0118
    Epoch 20/100
    1024/1024 [==============================] - 0s 92us/sample - loss: 0.0075 - val_loss: 0.0156
    Epoch 21/100
    1024/1024 [==============================] - 0s 92us/sample - loss: 0.0064 - val_loss: 0.0141
    Epoch 22/100
    1024/1024 [==============================] - 0s 92us/sample - loss: 0.0062 - val_loss: 0.0123
    Epoch 23/100
    1024/1024 [==============================] - 0s 87us/sample - loss: 0.0065 - val_loss: 0.0118
    Epoch 24/100
    1024/1024 [==============================] - 0s 92us/sample - loss: 0.0057 - val_loss: 0.0120
    Epoch 25/100
    1024/1024 [==============================] - 0s 93us/sample - loss: 0.0065 - val_loss: 0.0128
    Epoch 26/100
    1024/1024 [==============================] - 0s 98us/sample - loss: 0.0061 - val_loss: 0.0152
    Epoch 27/100
    1024/1024 [==============================] - 0s 94us/sample - loss: 0.0062 - val_loss: 0.0114
    Epoch 28/100
    1024/1024 [==============================] - 0s 90us/sample - loss: 0.0061 - val_loss: 0.0151
    Epoch 29/100
    1024/1024 [==============================] - 0s 91us/sample - loss: 0.0058 - val_loss: 0.0118
    Epoch 30/100
    1024/1024 [==============================] - 0s 90us/sample - loss: 0.0058 - val_loss: 0.0132
    Epoch 31/100
    1024/1024 [==============================] - 0s 84us/sample - loss: 0.0058 - val_loss: 0.0090
    Epoch 32/100
    1024/1024 [==============================] - 0s 83us/sample - loss: 0.0052 - val_loss: 0.0127
    Epoch 33/100
    1024/1024 [==============================] - 0s 86us/sample - loss: 0.0057 - val_loss: 0.0120
    Epoch 34/100
    1024/1024 [==============================] - 0s 82us/sample - loss: 0.0056 - val_loss: 0.0138
    Epoch 35/100
    1024/1024 [==============================] - 0s 83us/sample - loss: 0.0052 - val_loss: 0.0132
    Epoch 36/100
    1024/1024 [==============================] - 0s 84us/sample - loss: 0.0050 - val_loss: 0.0151
    Epoch 37/100
    1024/1024 [==============================] - 0s 83us/sample - loss: 0.0050 - val_loss: 0.0125
    Epoch 38/100
    1024/1024 [==============================] - 0s 82us/sample - loss: 0.0048 - val_loss: 0.0155
    Epoch 39/100
    1024/1024 [==============================] - 0s 82us/sample - loss: 0.0051 - val_loss: 0.0079
    Epoch 40/100
    1024/1024 [==============================] - 0s 85us/sample - loss: 0.0048 - val_loss: 0.0139
    Epoch 41/100
    1024/1024 [==============================] - 0s 83us/sample - loss: 0.0052 - val_loss: 0.0131
    Epoch 42/100
    1024/1024 [==============================] - 0s 81us/sample - loss: 0.0045 - val_loss: 0.0130
    Epoch 43/100
    1024/1024 [==============================] - 0s 83us/sample - loss: 0.0043 - val_loss: 0.0136
    Epoch 44/100
    1024/1024 [==============================] - 0s 84us/sample - loss: 0.0048 - val_loss: 0.0117
    Epoch 45/100
    1024/1024 [==============================] - 0s 84us/sample - loss: 0.0045 - val_loss: 0.0116
    Epoch 46/100
    1024/1024 [==============================] - 0s 91us/sample - loss: 0.0047 - val_loss: 0.0105
    Epoch 47/100
    1024/1024 [==============================] - 0s 91us/sample - loss: 0.0048 - val_loss: 0.0112
    Epoch 48/100
    1024/1024 [==============================] - 0s 89us/sample - loss: 0.0045 - val_loss: 0.0124
    Epoch 49/100
    1024/1024 [==============================] - 0s 90us/sample - loss: 0.0044 - val_loss: 0.0087
    Epoch 50/100
    1024/1024 [==============================] - 0s 89us/sample - loss: 0.0046 - val_loss: 0.0095
    Epoch 51/100
    1024/1024 [==============================] - 0s 89us/sample - loss: 0.0046 - val_loss: 0.0170
    Epoch 52/100
    1024/1024 [==============================] - 0s 89us/sample - loss: 0.0049 - val_loss: 0.0085
    Epoch 53/100
    1024/1024 [==============================] - 0s 88us/sample - loss: 0.0048 - val_loss: 0.0126
    Epoch 54/100
    1024/1024 [==============================] - 0s 91us/sample - loss: 0.0046 - val_loss: 0.0134
    Epoch 55/100
    1024/1024 [==============================] - 0s 92us/sample - loss: 0.0044 - val_loss: 0.0101
    Epoch 56/100
    1024/1024 [==============================] - 0s 90us/sample - loss: 0.0040 - val_loss: 0.0127
    Epoch 57/100
    1024/1024 [==============================] - 0s 90us/sample - loss: 0.0038 - val_loss: 0.0128
    Epoch 58/100
    1024/1024 [==============================] - 0s 90us/sample - loss: 0.0039 - val_loss: 0.0104
    Epoch 59/100
    1024/1024 [==============================] - 0s 89us/sample - loss: 0.0039 - val_loss: 0.0151
    Epoch 60/100
    1024/1024 [==============================] - 0s 90us/sample - loss: 0.0039 - val_loss: 0.0116
    Epoch 61/100
    1024/1024 [==============================] - 0s 89us/sample - loss: 0.0042 - val_loss: 0.0146
    Epoch 62/100
    1024/1024 [==============================] - 0s 89us/sample - loss: 0.0046 - val_loss: 0.0085
    Epoch 63/100
    1024/1024 [==============================] - 0s 88us/sample - loss: 0.0040 - val_loss: 0.0132
    Epoch 64/100
    1024/1024 [==============================] - 0s 88us/sample - loss: 0.0042 - val_loss: 0.0119
    Epoch 65/100
    1024/1024 [==============================] - 0s 89us/sample - loss: 0.0038 - val_loss: 0.0101
    Epoch 66/100
    1024/1024 [==============================] - 0s 89us/sample - loss: 0.0037 - val_loss: 0.0104
    Epoch 67/100
    1024/1024 [==============================] - 0s 88us/sample - loss: 0.0038 - val_loss: 0.0128
    Epoch 68/100
    1024/1024 [==============================] - 0s 88us/sample - loss: 0.0038 - val_loss: 0.0162
    Epoch 69/100
    1024/1024 [==============================] - 0s 90us/sample - loss: 0.0035 - val_loss: 0.0122
    Epoch 70/100
    1024/1024 [==============================] - 0s 90us/sample - loss: 0.0038 - val_loss: 0.0107
    Epoch 71/100
    1024/1024 [==============================] - 0s 88us/sample - loss: 0.0035 - val_loss: 0.0130
    Epoch 72/100
    1024/1024 [==============================] - 0s 91us/sample - loss: 0.0038 - val_loss: 0.0101
    Epoch 73/100
    1024/1024 [==============================] - 0s 90us/sample - loss: 0.0035 - val_loss: 0.0113
    Epoch 74/100
    1024/1024 [==============================] - 0s 91us/sample - loss: 0.0035 - val_loss: 0.0111
    Epoch 75/100
    1024/1024 [==============================] - 0s 89us/sample - loss: 0.0038 - val_loss: 0.0124
    Epoch 76/100
    1024/1024 [==============================] - 0s 89us/sample - loss: 0.0031 - val_loss: 0.0103
    Epoch 77/100
    1024/1024 [==============================] - 0s 90us/sample - loss: 0.0035 - val_loss: 0.0151
    Epoch 78/100
    1024/1024 [==============================] - 0s 89us/sample - loss: 0.0038 - val_loss: 0.0112
    Epoch 79/100
    1024/1024 [==============================] - 0s 89us/sample - loss: 0.0037 - val_loss: 0.0147
    Epoch 80/100
    1024/1024 [==============================] - 0s 89us/sample - loss: 0.0036 - val_loss: 0.0132
    Epoch 81/100
    1024/1024 [==============================] - 0s 89us/sample - loss: 0.0037 - val_loss: 0.0103
    Epoch 82/100
    1024/1024 [==============================] - 0s 91us/sample - loss: 0.0038 - val_loss: 0.0106
    Epoch 83/100
    1024/1024 [==============================] - 0s 90us/sample - loss: 0.0034 - val_loss: 0.0113
    Epoch 84/100
    1024/1024 [==============================] - 0s 91us/sample - loss: 0.0038 - val_loss: 0.0091
    Epoch 85/100
    1024/1024 [==============================] - 0s 90us/sample - loss: 0.0035 - val_loss: 0.0159
    Epoch 86/100
    1024/1024 [==============================] - 0s 91us/sample - loss: 0.0034 - val_loss: 0.0087
    Epoch 87/100
    1024/1024 [==============================] - 0s 90us/sample - loss: 0.0035 - val_loss: 0.0114
    Epoch 88/100
    1024/1024 [==============================] - 0s 90us/sample - loss: 0.0033 - val_loss: 0.0083
    Epoch 89/100
    1024/1024 [==============================] - 0s 89us/sample - loss: 0.0034 - val_loss: 0.0117
    Epoch 90/100
    1024/1024 [==============================] - 0s 88us/sample - loss: 0.0034 - val_loss: 0.0096
    Epoch 91/100
    1024/1024 [==============================] - 0s 90us/sample - loss: 0.0037 - val_loss: 0.0087
    Epoch 92/100
    1024/1024 [==============================] - 0s 89us/sample - loss: 0.0032 - val_loss: 0.0105
    Epoch 93/100
    1024/1024 [==============================] - 0s 91us/sample - loss: 0.0035 - val_loss: 0.0122
    Epoch 94/100
    1024/1024 [==============================] - 0s 89us/sample - loss: 0.0032 - val_loss: 0.0104
    Epoch 95/100
    1024/1024 [==============================] - 0s 92us/sample - loss: 0.0030 - val_loss: 0.0091
    Epoch 96/100
    1024/1024 [==============================] - 0s 91us/sample - loss: 0.0034 - val_loss: 0.0116
    Epoch 97/100
    1024/1024 [==============================] - 0s 90us/sample - loss: 0.0030 - val_loss: 0.0085
    Epoch 98/100
    1024/1024 [==============================] - 0s 92us/sample - loss: 0.0030 - val_loss: 0.0095
    Epoch 99/100
    1024/1024 [==============================] - 0s 91us/sample - loss: 0.0032 - val_loss: 0.0118
    Epoch 100/100
    1024/1024 [==============================] - 0s 89us/sample - loss: 0.0032 - val_loss: 0.0079



```python
# Plot training history
plot_training(history_cnn)
```


![png](images/output_77_0.png)



```python
# Plot 7-day model predictions for USA
plot_usa_last(cnn, 40)
```


![png](images/output_78_0.png)



```python
# Plot 7-day model predictions for target countries
plot_countries(cnn, response, ncols=3)
```

    Plotting 16 countries on 3x5 grid with 1 left out



![png](images/output_79_1.png)


#### Define Combination Network for Population Level Growth


```python
# Define callbacks
cb = [
    ReduceLROnPlateau(monitor = 'val_loss', patience = 10, verbose = 1, factor = 0.5),
    EarlyStopping(monitor = 'val_loss', patience = 20, restore_best_weights = True)
]
```


```python
# Define model
mul = Sequential(name = 'Multi_Model')

mul.add(Conv1D(filters = 64, kernel_size = 3, padding = 'valid', activation = 'relu',
               input_shape=(days_train, features)))
mul.add(Conv1D(filters = 128, kernel_size = 3, padding = 'valid', activation = 'relu'))
mul.add(LSTM(256, return_sequences = True, recurrent_dropout = 0.5))
mul.add(LSTM(128, recurrent_dropout = 0.4))

mul.add(Dense(64, activation = 'relu'))
mul.add(Dropout(0.3))
mul.add(Dense(32, activation = 'relu'))

mul.add(Dense(days_pred, activation = 'linear'))

mul.compile(optimizer=Adam(learning_rate = 0.001), loss='mse', metric = ['rmse'])

mul.summary()
```

    Model: "Multi_Model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv1d_2 (Conv1D)            (None, 12, 64)            1600      
    _________________________________________________________________
    conv1d_3 (Conv1D)            (None, 10, 128)           24704     
    _________________________________________________________________
    lstm_2 (LSTM)                (None, 10, 256)           394240    
    _________________________________________________________________
    lstm_3 (LSTM)                (None, 128)               197120    
    _________________________________________________________________
    dense_5 (Dense)              (None, 64)                8256      
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 64)                0         
    _________________________________________________________________
    dense_6 (Dense)              (None, 32)                2080      
    _________________________________________________________________
    dense_7 (Dense)              (None, 7)                 231       
    =================================================================
    Total params: 628,231
    Trainable params: 628,231
    Non-trainable params: 0
    _________________________________________________________________



```python
# Fit model to X/Y data
history_mul = mul.fit(train_x, train_y, epochs=100, verbose=1, callbacks = cb,
                      batch_size = 8, class_weight = cw, validation_split = 0.15)
```

    Train on 967 samples, validate on 171 samples
    Epoch 1/100
    967/967 [==============================] - 8s 8ms/sample - loss: 0.0231 - val_loss: 0.0078
    Epoch 2/100
    967/967 [==============================] - 4s 4ms/sample - loss: 0.0108 - val_loss: 0.0073
    Epoch 3/100
    967/967 [==============================] - 4s 4ms/sample - loss: 0.0104 - val_loss: 0.0184
    Epoch 4/100
    967/967 [==============================] - 4s 4ms/sample - loss: 0.0078 - val_loss: 0.0114
    Epoch 5/100
    967/967 [==============================] - 4s 4ms/sample - loss: 0.0063 - val_loss: 0.0061
    Epoch 6/100
    967/967 [==============================] - 4s 4ms/sample - loss: 0.0055 - val_loss: 0.0056
    Epoch 7/100
    967/967 [==============================] - 4s 4ms/sample - loss: 0.0052 - val_loss: 0.0067
    Epoch 8/100
    967/967 [==============================] - 4s 4ms/sample - loss: 0.0045 - val_loss: 0.0071
    Epoch 9/100
    967/967 [==============================] - 4s 4ms/sample - loss: 0.0041 - val_loss: 0.0071
    Epoch 10/100
    967/967 [==============================] - 4s 4ms/sample - loss: 0.0040 - val_loss: 0.0079
    Epoch 11/100
    967/967 [==============================] - 4s 4ms/sample - loss: 0.0041 - val_loss: 0.0065
    Epoch 12/100
    967/967 [==============================] - 4s 4ms/sample - loss: 0.0040 - val_loss: 0.0089
    Epoch 13/100
    967/967 [==============================] - 4s 4ms/sample - loss: 0.0031 - val_loss: 0.0063
    Epoch 14/100
    967/967 [==============================] - 4s 4ms/sample - loss: 0.0032 - val_loss: 0.0056
    Epoch 15/100
    967/967 [==============================] - 4s 4ms/sample - loss: 0.0027 - val_loss: 0.0080
    Epoch 16/100
    952/967 [============================>.] - ETA: 0s - loss: 0.0025
    Epoch 00016: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
    967/967 [==============================] - 4s 4ms/sample - loss: 0.0025 - val_loss: 0.0095
    Epoch 17/100
    967/967 [==============================] - 4s 4ms/sample - loss: 0.0025 - val_loss: 0.0082
    Epoch 18/100
    967/967 [==============================] - 4s 4ms/sample - loss: 0.0020 - val_loss: 0.0076
    Epoch 19/100
    967/967 [==============================] - 4s 4ms/sample - loss: 0.0018 - val_loss: 0.0086
    Epoch 20/100
    967/967 [==============================] - 4s 4ms/sample - loss: 0.0018 - val_loss: 0.0082
    Epoch 21/100
    967/967 [==============================] - 4s 4ms/sample - loss: 0.0020 - val_loss: 0.0078
    Epoch 22/100
    967/967 [==============================] - 4s 4ms/sample - loss: 0.0019 - val_loss: 0.0083
    Epoch 23/100
    967/967 [==============================] - 4s 4ms/sample - loss: 0.0021 - val_loss: 0.0070
    Epoch 24/100
    967/967 [==============================] - 4s 4ms/sample - loss: 0.0020 - val_loss: 0.0096
    Epoch 25/100
    967/967 [==============================] - 4s 4ms/sample - loss: 0.0018 - val_loss: 0.0086
    Epoch 26/100
    952/967 [============================>.] - ETA: 0s - loss: 0.0016
    Epoch 00026: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.
    967/967 [==============================] - 4s 4ms/sample - loss: 0.0016 - val_loss: 0.0075
    Epoch 27/100
    967/967 [==============================] - 4s 4ms/sample - loss: 0.0016 - val_loss: 0.0080
    Epoch 28/100
    967/967 [==============================] - 4s 4ms/sample - loss: 0.0016 - val_loss: 0.0078
    Epoch 29/100
    967/967 [==============================] - 4s 4ms/sample - loss: 0.0015 - val_loss: 0.0082
    Epoch 30/100
    967/967 [==============================] - 4s 4ms/sample - loss: 0.0014 - val_loss: 0.0072
    Epoch 31/100
    967/967 [==============================] - 4s 4ms/sample - loss: 0.0015 - val_loss: 0.0088
    Epoch 32/100
    967/967 [==============================] - 4s 4ms/sample - loss: 0.0014 - val_loss: 0.0082
    Epoch 33/100
    967/967 [==============================] - 4s 4ms/sample - loss: 0.0014 - val_loss: 0.0091
    Epoch 34/100
    967/967 [==============================] - 4s 4ms/sample - loss: 0.0013 - val_loss: 0.0106



```python
# Plot training history
plot_training(history_mul)
```


![png](images/output_84_0.png)



```python
# Plot 7-day model predictions for USA
plot_usa_last(mul, 40)
```


![png](images/output_85_0.png)



```python
# Plot 7-day model predictions for target countries
plot_countries(mul, response, ncols=3)
```

    Plotting 16 countries on 3x5 grid with 1 left out



![png](images/output_86_1.png)


## Generalizable COVID-19 Model Conclusion

We achieve the best fit curves and loss using a combination model of CNN and LSTM, with dropout to the recurrent activations of the LSTM layers and the Dense layers towards the end. We train on 14-day windows of data, predicting 7 days ahead. The network learns and can reasonably reproduce a number of differently shaped growth curves, including those with multiple local maxima.

The model learns to map 14-day histories to 7-day predictions into the future based on a country's case, testing, and mobility data, but with no explicit mapping to specific countries. It is a general model meant to make predictions for 7 days based on the preceding 2 weeks, and thus tries to learn how these relationships generally unfold. The plotted predictions therefore do not rely on those of the preceding sequence, as they are independent, hence these lines look somewhat jagged; every round of prediction is made without other context than the preceding 14 days of inputs. This can be improved in future work, but in this case our goal was to make a general country-agnostic and history-agnostic predictor. It is therefore interesting how closely it generally follows the curves, and the extent to which it seems able to anticipate spikes.

It also appears that the model is successfully able to alert to undiagnosed case load increases. In Norway at day 10, Canada at day 60, Sweden at day 60 and the Netherlands at day 20, the model suggests a higher than reported number of new cases per 100k citizens that is subsequently validated by dramatic spikes within 5-10 days after that. Judging by the shape of the data, these almost look like the government caught up on belated or missing measurements.

One counter example may be Germany, where the model appears to be predicting continual rapid growth when the curve is really peaking. We think this is actual a correct prediction contrary to that small but sudden drop in Germany's rising curve, where our model predicted continued growth in spite of the drop. Following that, the model adjusts and almost perfectly predicts Germany's peak.

In countries like Italy, Switzerland, France and Spain, the model does very well aligning with those nation's peaks and keeping ahead of the decline in caseload. The United States remains a problematic outlier. Because the country plateaued in its efforts to contain and combat the virus during April, there has been a persistent and stable new case growth for multiple weeks. This trend goes against that experienced by nearly every other impacted nation and makes a generalizable model extremely difficult to apply to the American experience.

This appears to also be occurring in the United Kingdom, which within the past 5 days is seeing a separation between growth expected by other nations and the case growth observed. In this scenario, our recommendation would be to use this model to identify outlier nations who continue to experience difficulty as a way of flagging those whose response is not yet sufficient or who were successfully on trend but may have elected to re-open too soon. The fact that the model continues to predict a declining caseload for the UK means the UK has more cases than countries with comparable 14-day histories of case, test, and mobility data.

------

--------

# Predicting a Country's Case Growth

We next turn to the challenge of accurately predicting a target country's case or death growth using a variety of methods. We restructure our alternate time series before running a number of experiments to determine the optimum approach and training data

#### Define data structuring helper functions


```python
# Define function to generate multivariate time series dataframe
def generate_multivariate_df(dataframe, rollup_column: str, metric_column: str, country_column='country', fillna=False):
    # Generate list of unique countries to iterate through
    unique_countries_in_df = dataframe[country_column].unique()

    # Create starting series with index as unique values from dataframe
    base_series = pd.Series(index=np.sort(dataframe[rollup_column].unique()))

    # Iterate through countries and add time series to base_series
    for country in unique_countries_in_df:
        country_series = dataframe[dataframe[country_column] == country].groupby(rollup_column)[metric_column].mean().rename(country)
        base_series = pd.concat([base_series, country_series], axis=1, sort=False)
    base_series = base_series.drop(0, axis=1)
    if fillna == True:
        base_series = base_series.fillna(0)

    return base_series
```


```python
# Define function to filter dataframe to target countries
def filter_to_target_countries(dataframe):

    return dataframe[dataframe.columns[dataframe.columns.isin(target_countries)]]
```

### Structure testing data


```python
# Create OurWorldInData testing by date dataframe
testing = generate_multivariate_df(testing_raw, 'date', 'tests', fillna=False)

testing = standardize_country_names(testing)

testing = filter_to_target_countries(testing)
```


```python
## Some countries only report testing numbers on a weekly basis.
## For those, we forward fill their weekly reported number for each subsequent day until a new number is reported.
## These means we have a step curve, but it helps standardize time series for smoother learning.

# Forward fill recently missing date for tests
print('Original missing tests:')
missing_original = testing.isna().sum()

# Within each country, forward-fill after first non-missing total tests
testing = testing.transform(lambda x: x.ffill())
print('After forward-filling:')
missing_ff = testing.isna().sum()

# Then impute 0 for any remaining missing tests, as that will be because testing was not yet performed
testing.fillna(0, inplace = True)
print('After imputing zeros:')
missing_zeros = testing.isna().sum()

missing_nums = pd.concat([
    missing_original.rename('Original'),
    missing_ff.rename('Forward Fill'),
    missing_zeros.rename('Impute Zeros and End')
], axis=1)
display(missing_nums)
```

    Original missing tests:
    After forward-filling:
    After imputing zeros:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Original</th>
      <th>Forward Fill</th>
      <th>Impute Zeros and End</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Austria</td>
      <td>57</td>
      <td>55</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Belgium</td>
      <td>62</td>
      <td>60</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Canada</td>
      <td>79</td>
      <td>77</td>
      <td>0</td>
    </tr>
    <tr>
      <td>France</td>
      <td>107</td>
      <td>54</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Germany</td>
      <td>122</td>
      <td>67</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Italy</td>
      <td>55</td>
      <td>54</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Netherlands</td>
      <td>119</td>
      <td>74</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Norway</td>
      <td>83</td>
      <td>75</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Portugal</td>
      <td>65</td>
      <td>60</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Singapore</td>
      <td>126</td>
      <td>97</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Spain</td>
      <td>127</td>
      <td>103</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Sweden</td>
      <td>121</td>
      <td>60</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Switzerland</td>
      <td>28</td>
      <td>23</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Turkey</td>
      <td>79</td>
      <td>77</td>
      <td>0</td>
    </tr>
    <tr>
      <td>United_Kingdom</td>
      <td>97</td>
      <td>96</td>
      <td>0</td>
    </tr>
    <tr>
      <td>United_States</td>
      <td>67</td>
      <td>66</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Display preview of table
testing.tail(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Austria</th>
      <th>Belgium</th>
      <th>Canada</th>
      <th>France</th>
      <th>Germany</th>
      <th>Italy</th>
      <th>Netherlands</th>
      <th>Norway</th>
      <th>Portugal</th>
      <th>Singapore</th>
      <th>Spain</th>
      <th>Sweden</th>
      <th>Switzerland</th>
      <th>Turkey</th>
      <th>United_Kingdom</th>
      <th>United_States</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2020-05-08</td>
      <td>304069.0</td>
      <td>461303.0</td>
      <td>1032012.0</td>
      <td>831174.0</td>
      <td>2755770.0</td>
      <td>2445063.0</td>
      <td>254456.0</td>
      <td>195921.0</td>
      <td>485925.0</td>
      <td>175604.0</td>
      <td>1625211.0</td>
      <td>148500.0</td>
      <td>309206.0</td>
      <td>1298806.0</td>
      <td>1631561.0</td>
      <td>8408788.0</td>
    </tr>
    <tr>
      <td>2020-05-09</td>
      <td>311690.0</td>
      <td>461303.0</td>
      <td>1067595.0</td>
      <td>831174.0</td>
      <td>2755770.0</td>
      <td>2514234.0</td>
      <td>254456.0</td>
      <td>195921.0</td>
      <td>485925.0</td>
      <td>175604.0</td>
      <td>1625211.0</td>
      <td>148500.0</td>
      <td>309595.0</td>
      <td>1334411.0</td>
      <td>1728443.0</td>
      <td>8709630.0</td>
    </tr>
    <tr>
      <td>2020-05-10</td>
      <td>316508.0</td>
      <td>461303.0</td>
      <td>1071379.0</td>
      <td>831174.0</td>
      <td>2755770.0</td>
      <td>2514234.0</td>
      <td>254456.0</td>
      <td>195921.0</td>
      <td>485925.0</td>
      <td>175604.0</td>
      <td>1625211.0</td>
      <td>148500.0</td>
      <td>309595.0</td>
      <td>1334411.0</td>
      <td>1728443.0</td>
      <td>8709630.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot US testing numbers
plt.figure(figsize=(20,8))
for i in range(len(testing.columns)):
    plt.plot(testing.iloc[60:-5, i], label=testing.columns[i])
plt.legend(fontsize=15)
plt.title("Tests performed in target countries", fontsize=15)
plt.xticks(rotation=45)
plt.show()
```


![png](images/output_98_0.png)


### Structure Apple mobility data


```python
# Create Apple Mobility by date dataframe
apple_overall = round(np.transpose(apple_raw.groupby('country').agg('mean')), 3)

apple_overall = standardize_country_names(apple_overall)

apple_overall = filter_to_target_countries(apple_overall)

apple_overall.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>country</th>
      <th>Austria</th>
      <th>Belgium</th>
      <th>Canada</th>
      <th>France</th>
      <th>Germany</th>
      <th>Italy</th>
      <th>Netherlands</th>
      <th>Norway</th>
      <th>Portugal</th>
      <th>Singapore</th>
      <th>Spain</th>
      <th>Sweden</th>
      <th>Switzerland</th>
      <th>Turkey</th>
      <th>United_Kingdom</th>
      <th>United_States</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2020-05-04</td>
      <td>64.350</td>
      <td>51.920</td>
      <td>53.397</td>
      <td>27.950</td>
      <td>65.910</td>
      <td>32.887</td>
      <td>52.050</td>
      <td>80.397</td>
      <td>33.060</td>
      <td>26.890</td>
      <td>28.693</td>
      <td>80.630</td>
      <td>66.893</td>
      <td>50.785</td>
      <td>35.853</td>
      <td>57.093</td>
    </tr>
    <tr>
      <td>2020-05-05</td>
      <td>63.555</td>
      <td>53.990</td>
      <td>57.403</td>
      <td>29.433</td>
      <td>67.893</td>
      <td>33.463</td>
      <td>55.383</td>
      <td>80.463</td>
      <td>35.355</td>
      <td>27.710</td>
      <td>29.203</td>
      <td>83.023</td>
      <td>62.747</td>
      <td>51.225</td>
      <td>37.353</td>
      <td>58.380</td>
    </tr>
    <tr>
      <td>2020-05-06</td>
      <td>67.580</td>
      <td>58.030</td>
      <td>59.160</td>
      <td>31.250</td>
      <td>70.387</td>
      <td>33.633</td>
      <td>57.353</td>
      <td>83.473</td>
      <td>35.595</td>
      <td>27.037</td>
      <td>29.270</td>
      <td>85.540</td>
      <td>68.617</td>
      <td>51.760</td>
      <td>39.023</td>
      <td>58.420</td>
    </tr>
    <tr>
      <td>2020-05-07</td>
      <td>71.145</td>
      <td>59.153</td>
      <td>57.887</td>
      <td>31.980</td>
      <td>72.647</td>
      <td>34.437</td>
      <td>59.143</td>
      <td>82.640</td>
      <td>36.390</td>
      <td>29.757</td>
      <td>28.357</td>
      <td>87.870</td>
      <td>72.363</td>
      <td>52.185</td>
      <td>39.627</td>
      <td>61.623</td>
    </tr>
    <tr>
      <td>2020-05-08</td>
      <td>73.565</td>
      <td>60.973</td>
      <td>61.907</td>
      <td>27.453</td>
      <td>75.377</td>
      <td>35.737</td>
      <td>63.570</td>
      <td>86.350</td>
      <td>38.315</td>
      <td>30.830</td>
      <td>28.873</td>
      <td>94.350</td>
      <td>75.027</td>
      <td>52.025</td>
      <td>38.833</td>
      <td>66.400</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot mobility numbers
plt.figure(figsize=(20,8))
for i in range(len(apple_overall.columns)):
    plt.plot(apple_overall.iloc[:, i], label=apple_overall.columns[i])
plt.legend(fontsize=8)
plt.title("Apple Mobility Trend in target countries", fontsize=15)
plt.xticks("")
plt.show()
```


![png](images/output_101_0.png)


### Structure Google mobility data


```python
# Create Google Mobility by date dataframe
google_overall = generate_multivariate_df(google_raw, 'date', 'overall_change_from_baseline', fillna=False)

google_overall = standardize_country_names(google_overall)

google_overall = filter_to_target_countries(google_overall)

google_overall.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Austria</th>
      <th>Belgium</th>
      <th>Canada</th>
      <th>Switzerland</th>
      <th>Germany</th>
      <th>Spain</th>
      <th>France</th>
      <th>United_Kingdom</th>
      <th>Italy</th>
      <th>Netherlands</th>
      <th>Norway</th>
      <th>Portugal</th>
      <th>Sweden</th>
      <th>Singapore</th>
      <th>Turkey</th>
      <th>United_States</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2020-04-28</td>
      <td>-22.9234</td>
      <td>-40.33325</td>
      <td>-26.785786</td>
      <td>-27.758667</td>
      <td>-15.539235</td>
      <td>-51.82500</td>
      <td>-45.654714</td>
      <td>-42.187704</td>
      <td>-47.333381</td>
      <td>-20.858923</td>
      <td>-3.652750</td>
      <td>-38.363524</td>
      <td>-8.261318</td>
      <td>-41.000</td>
      <td>-27.500</td>
      <td>-21.872843</td>
    </tr>
    <tr>
      <td>2020-04-29</td>
      <td>-31.7034</td>
      <td>-37.83350</td>
      <td>-23.565500</td>
      <td>-25.465407</td>
      <td>-15.774529</td>
      <td>-50.62825</td>
      <td>-45.464429</td>
      <td>-40.649020</td>
      <td>-45.976238</td>
      <td>-15.487308</td>
      <td>-0.486250</td>
      <td>-39.231762</td>
      <td>-4.870455</td>
      <td>-42.333</td>
      <td>-27.500</td>
      <td>-21.705272</td>
    </tr>
    <tr>
      <td>2020-04-30</td>
      <td>-19.3100</td>
      <td>-33.08300</td>
      <td>-23.678643</td>
      <td>-29.416111</td>
      <td>-13.362765</td>
      <td>-48.40170</td>
      <td>-42.333357</td>
      <td>-40.145086</td>
      <td>-42.722238</td>
      <td>-17.397462</td>
      <td>6.722333</td>
      <td>-34.717476</td>
      <td>-3.200773</td>
      <td>-42.333</td>
      <td>-21.000</td>
      <td>-21.579472</td>
    </tr>
    <tr>
      <td>2020-05-01</td>
      <td>-47.3532</td>
      <td>-52.16650</td>
      <td>-24.178571</td>
      <td>-39.454333</td>
      <td>-45.901941</td>
      <td>-66.53000</td>
      <td>-63.214357</td>
      <td>-38.290645</td>
      <td>-65.117571</td>
      <td>-21.051231</td>
      <td>-22.222250</td>
      <td>-55.710286</td>
      <td>-31.125818</td>
      <td>-43.333</td>
      <td>-58.833</td>
      <td>-18.734739</td>
    </tr>
    <tr>
      <td>2020-05-02</td>
      <td>-17.9365</td>
      <td>-35.66675</td>
      <td>-21.480929</td>
      <td>-31.025000</td>
      <td>-17.029412</td>
      <td>-42.79165</td>
      <td>-45.523714</td>
      <td>-40.674533</td>
      <td>-47.846095</td>
      <td>-14.948923</td>
      <td>10.597167</td>
      <td>-43.756333</td>
      <td>-3.243136</td>
      <td>-41.000</td>
      <td>-63.167</td>
      <td>-11.273210</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot mobility numbers
plt.figure(figsize=(20,8))
for i in range(len(google_overall.columns)):
    plt.plot(google_overall.iloc[:, i], label=google_overall.columns[i])
plt.legend(fontsize=10)
plt.title("Google Mobility Trend in target countries", fontsize=15)
plt.xticks("")
plt.show()
```


![png](images/output_104_0.png)


### Structure case data for supervised learning


```python
# Define starting response column
response = 'new_cases_smooth'
```

#### Create date-indexed case dataframe


```python
cases_date = generate_multivariate_df(ecdc_country, 'date', response, fillna=True)

cases_date = standardize_country_names(cases_date)

cases_date = filter_to_target_countries(cases_date)

cases_date.tail(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Austria</th>
      <th>Belgium</th>
      <th>Canada</th>
      <th>France</th>
      <th>Germany</th>
      <th>Italy</th>
      <th>Netherlands</th>
      <th>Norway</th>
      <th>Portugal</th>
      <th>Singapore</th>
      <th>Spain</th>
      <th>Sweden</th>
      <th>Switzerland</th>
      <th>Turkey</th>
      <th>United_Kingdom</th>
      <th>United_States</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2020-05-08</td>
      <td>40.6</td>
      <td>380.6</td>
      <td>1641.6</td>
      <td>1360.0</td>
      <td>960.8</td>
      <td>1306.0</td>
      <td>307.6</td>
      <td>47.2</td>
      <td>305.0</td>
      <td>678.2</td>
      <td>1078.2</td>
      <td>508.2</td>
      <td>61.8</td>
      <td>1869.2</td>
      <td>4891.0</td>
      <td>24780.6</td>
    </tr>
    <tr>
      <td>2020-05-09</td>
      <td>39.4</td>
      <td>421.0</td>
      <td>1392.0</td>
      <td>1426.8</td>
      <td>1075.2</td>
      <td>1293.6</td>
      <td>304.4</td>
      <td>45.0</td>
      <td>348.8</td>
      <td>700.4</td>
      <td>1113.4</td>
      <td>589.6</td>
      <td>60.4</td>
      <td>1904.8</td>
      <td>4953.0</td>
      <td>25177.6</td>
    </tr>
    <tr>
      <td>2020-05-10</td>
      <td>41.6</td>
      <td>465.8</td>
      <td>1386.0</td>
      <td>1398.2</td>
      <td>1071.6</td>
      <td>1266.0</td>
      <td>322.4</td>
      <td>44.4</td>
      <td>376.4</td>
      <td>736.4</td>
      <td>0.0</td>
      <td>640.0</td>
      <td>54.0</td>
      <td>1891.2</td>
      <td>4935.2</td>
      <td>25781.4</td>
    </tr>
  </tbody>
</table>
</div>



#### Create Integrated Dataframe


```python
# Define function to join response dataframe with additional variables
def enrich_dataframe(dataframe):
    dataframe = dataframe.join(testing, how='outer', lsuffix='_response', rsuffix='_tests')
    dataframe = dataframe.join(apple_overall, how='outer')
    dataframe = dataframe.join(google_overall, how='outer', lsuffix='_apple_mobility', rsuffix='_google_mobility')
    return dataframe
```


```python
# Enrich cases by date
covid_by_date = enrich_dataframe(cases_date)
covid_by_date.tail(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Austria_response</th>
      <th>Belgium_response</th>
      <th>Canada_response</th>
      <th>France_response</th>
      <th>Germany_response</th>
      <th>Italy_response</th>
      <th>Netherlands_response</th>
      <th>Norway_response</th>
      <th>Portugal_response</th>
      <th>Singapore_response</th>
      <th>...</th>
      <th>France_google_mobility</th>
      <th>United_Kingdom_google_mobility</th>
      <th>Italy_google_mobility</th>
      <th>Netherlands_google_mobility</th>
      <th>Norway_google_mobility</th>
      <th>Portugal_google_mobility</th>
      <th>Sweden_google_mobility</th>
      <th>Singapore_google_mobility</th>
      <th>Turkey_google_mobility</th>
      <th>United_States_google_mobility</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2020-05-08</td>
      <td>40.6</td>
      <td>380.6</td>
      <td>1641.6</td>
      <td>1360.0</td>
      <td>960.8</td>
      <td>1306.0</td>
      <td>307.6</td>
      <td>47.2</td>
      <td>305.0</td>
      <td>678.2</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2020-05-09</td>
      <td>39.4</td>
      <td>421.0</td>
      <td>1392.0</td>
      <td>1426.8</td>
      <td>1075.2</td>
      <td>1293.6</td>
      <td>304.4</td>
      <td>45.0</td>
      <td>348.8</td>
      <td>700.4</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2020-05-10</td>
      <td>41.6</td>
      <td>465.8</td>
      <td>1386.0</td>
      <td>1398.2</td>
      <td>1071.6</td>
      <td>1266.0</td>
      <td>322.4</td>
      <td>44.4</td>
      <td>376.4</td>
      <td>736.4</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 64 columns</p>
</div>



#### Explore time series range limits


```python
# Confirm range of nulls on mobility
print("Apple:")
print(min(covid_by_date[covid_by_date['United_States_apple_mobility'].isna() == False].index.date),
" => ", max(covid_by_date[covid_by_date['United_States_apple_mobility'].isna() == False].index.date))
print("Google:")
print(min(covid_by_date[covid_by_date['United_States_google_mobility'].isna() == False].index.date),
" => ", max(covid_by_date[covid_by_date['United_States_google_mobility'].isna() == False].index.date))
```

    Apple:
    2020-01-13  =>  2020-05-08
    Google:
    2020-02-15  =>  2020-05-02



```python
# Mobility numbers are updated at a less frequent pace than cases
# To manage a few days delay on mobility, we forward-fill the last mobility number

# Percent null values
null_pcts_pre = round(covid_by_date.isna().sum()/len(covid_by_date)*100, 1)

# Within each country, forward-fill after first non-missing total tests
covid_by_date = covid_by_date.transform(lambda x: x.ffill())
null_pcts_post = round(covid_by_date.isna().sum()/len(covid_by_date)*100, 1)

null_pcts = pd.concat([null_pcts_pre.rename('Pre'), null_pcts_post.rename('Post')], axis=1)
display(null_pcts[(null_pcts['Pre'] > 0) & (null_pcts.index.str.contains('Norway'))])
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pre</th>
      <th>Post</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Norway_tests</td>
      <td>0.8</td>
      <td>0.8</td>
    </tr>
    <tr>
      <td>Norway_apple_mobility</td>
      <td>11.4</td>
      <td>9.8</td>
    </tr>
    <tr>
      <td>Norway_google_mobility</td>
      <td>40.9</td>
      <td>34.8</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Define function to filter dataframe to specific range
def filter_range(dataframe, start, end):
    return dataframe[(dataframe.index >= start) & (dataframe.index <= end)]
```


```python
# Filter to non-null range since unable to impute mobility
covid_ranged = filter_range(covid_by_date, '2020-02-15', '2020-05-02')
display(covid_ranged.head(3))
display(covid_ranged.tail(3))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Austria_response</th>
      <th>Belgium_response</th>
      <th>Canada_response</th>
      <th>France_response</th>
      <th>Germany_response</th>
      <th>Italy_response</th>
      <th>Netherlands_response</th>
      <th>Norway_response</th>
      <th>Portugal_response</th>
      <th>Singapore_response</th>
      <th>...</th>
      <th>France_google_mobility</th>
      <th>United_Kingdom_google_mobility</th>
      <th>Italy_google_mobility</th>
      <th>Netherlands_google_mobility</th>
      <th>Norway_google_mobility</th>
      <th>Portugal_google_mobility</th>
      <th>Sweden_google_mobility</th>
      <th>Singapore_google_mobility</th>
      <th>Turkey_google_mobility</th>
      <th>United_States_google_mobility</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2020-02-15</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.8</td>
      <td>...</td>
      <td>6.702429</td>
      <td>-11.410125</td>
      <td>7.047524</td>
      <td>3.051308</td>
      <td>7.194500</td>
      <td>6.161143</td>
      <td>1.997000</td>
      <td>-7.667</td>
      <td>-0.333</td>
      <td>3.779464</td>
    </tr>
    <tr>
      <td>2020-02-16</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.4</td>
      <td>...</td>
      <td>5.452429</td>
      <td>-7.373483</td>
      <td>8.611190</td>
      <td>-9.346154</td>
      <td>-3.277833</td>
      <td>3.061952</td>
      <td>-11.300045</td>
      <td>-12.500</td>
      <td>5.667</td>
      <td>4.247112</td>
    </tr>
    <tr>
      <td>2020-02-17</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.2</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.6</td>
      <td>...</td>
      <td>1.357143</td>
      <td>2.725007</td>
      <td>2.111190</td>
      <td>-0.307692</td>
      <td>1.402833</td>
      <td>2.815810</td>
      <td>-2.756000</td>
      <td>-6.000</td>
      <td>5.333</td>
      <td>-3.342209</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 64 columns</p>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Austria_response</th>
      <th>Belgium_response</th>
      <th>Canada_response</th>
      <th>France_response</th>
      <th>Germany_response</th>
      <th>Italy_response</th>
      <th>Netherlands_response</th>
      <th>Norway_response</th>
      <th>Portugal_response</th>
      <th>Singapore_response</th>
      <th>...</th>
      <th>France_google_mobility</th>
      <th>United_Kingdom_google_mobility</th>
      <th>Italy_google_mobility</th>
      <th>Netherlands_google_mobility</th>
      <th>Norway_google_mobility</th>
      <th>Portugal_google_mobility</th>
      <th>Sweden_google_mobility</th>
      <th>Singapore_google_mobility</th>
      <th>Turkey_google_mobility</th>
      <th>United_States_google_mobility</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2020-04-30</td>
      <td>59.2</td>
      <td>713.2</td>
      <td>1542.0</td>
      <td>1173.0</td>
      <td>1336.2</td>
      <td>2119.4</td>
      <td>453.4</td>
      <td>51.8</td>
      <td>284.2</td>
      <td>892.6</td>
      <td>...</td>
      <td>-42.333357</td>
      <td>-40.145086</td>
      <td>-42.722238</td>
      <td>-17.397462</td>
      <td>6.722333</td>
      <td>-34.717476</td>
      <td>-3.200773</td>
      <td>-42.333</td>
      <td>-21.000</td>
      <td>-21.579472</td>
    </tr>
    <tr>
      <td>2020-05-01</td>
      <td>58.0</td>
      <td>638.8</td>
      <td>1579.0</td>
      <td>1093.4</td>
      <td>988.8</td>
      <td>2022.4</td>
      <td>425.2</td>
      <td>48.6</td>
      <td>260.8</td>
      <td>695.2</td>
      <td>...</td>
      <td>-63.214357</td>
      <td>-38.290645</td>
      <td>-65.117571</td>
      <td>-21.051231</td>
      <td>-22.222250</td>
      <td>-55.710286</td>
      <td>-31.125818</td>
      <td>-43.333</td>
      <td>-58.833</td>
      <td>-18.734739</td>
    </tr>
    <tr>
      <td>2020-05-02</td>
      <td>53.8</td>
      <td>579.6</td>
      <td>1635.4</td>
      <td>1122.0</td>
      <td>1302.0</td>
      <td>1950.6</td>
      <td>389.2</td>
      <td>50.8</td>
      <td>301.0</td>
      <td>695.4</td>
      <td>...</td>
      <td>-45.523714</td>
      <td>-40.674533</td>
      <td>-47.846095</td>
      <td>-14.948923</td>
      <td>10.597167</td>
      <td>-43.756333</td>
      <td>-3.243136</td>
      <td>-41.000</td>
      <td>-63.167</td>
      <td>-11.273210</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 64 columns</p>
</div>



```python
# Apply MinMax scaling
covid_scaled = scale(covid_ranged)
display(covid_scaled.head(3))
display(covid_scaled.tail(3))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Austria_response</th>
      <th>Belgium_response</th>
      <th>Canada_response</th>
      <th>France_response</th>
      <th>Germany_response</th>
      <th>Italy_response</th>
      <th>Netherlands_response</th>
      <th>Norway_response</th>
      <th>Portugal_response</th>
      <th>Singapore_response</th>
      <th>...</th>
      <th>France_google_mobility</th>
      <th>United_Kingdom_google_mobility</th>
      <th>Italy_google_mobility</th>
      <th>Netherlands_google_mobility</th>
      <th>Norway_google_mobility</th>
      <th>Portugal_google_mobility</th>
      <th>Sweden_google_mobility</th>
      <th>Singapore_google_mobility</th>
      <th>Turkey_google_mobility</th>
      <th>United_States_google_mobility</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2020-02-15</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00011</td>
      <td>0.000000</td>
      <td>0.000067</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.003475</td>
      <td>...</td>
      <td>0.967386</td>
      <td>0.725984</td>
      <td>0.957145</td>
      <td>0.856218</td>
      <td>0.927778</td>
      <td>0.817011</td>
      <td>0.883007</td>
      <td>0.859841</td>
      <td>0.764706</td>
      <td>0.824496</td>
    </tr>
    <tr>
      <td>2020-02-16</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00011</td>
      <td>0.000000</td>
      <td>0.000067</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.004054</td>
      <td>...</td>
      <td>0.950091</td>
      <td>0.791039</td>
      <td>0.976599</td>
      <td>0.584130</td>
      <td>0.705500</td>
      <td>0.782663</td>
      <td>0.528526</td>
      <td>0.750000</td>
      <td>0.835294</td>
      <td>0.833477</td>
    </tr>
    <tr>
      <td>2020-02-17</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00011</td>
      <td>0.000041</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.004247</td>
      <td>...</td>
      <td>0.893427</td>
      <td>0.953788</td>
      <td>0.895729</td>
      <td>0.782498</td>
      <td>0.804848</td>
      <td>0.779935</td>
      <td>0.756298</td>
      <td>0.897727</td>
      <td>0.831365</td>
      <td>0.687727</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 64 columns</p>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Austria_response</th>
      <th>Belgium_response</th>
      <th>Canada_response</th>
      <th>France_response</th>
      <th>Germany_response</th>
      <th>Italy_response</th>
      <th>Netherlands_response</th>
      <th>Norway_response</th>
      <th>Portugal_response</th>
      <th>Singapore_response</th>
      <th>...</th>
      <th>France_google_mobility</th>
      <th>United_Kingdom_google_mobility</th>
      <th>Italy_google_mobility</th>
      <th>Netherlands_google_mobility</th>
      <th>Norway_google_mobility</th>
      <th>Portugal_google_mobility</th>
      <th>Sweden_google_mobility</th>
      <th>Singapore_google_mobility</th>
      <th>Turkey_google_mobility</th>
      <th>United_States_google_mobility</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2020-04-30</td>
      <td>0.072799</td>
      <td>0.478465</td>
      <td>0.847253</td>
      <td>0.242716</td>
      <td>0.224187</td>
      <td>0.371238</td>
      <td>0.377393</td>
      <td>0.168620</td>
      <td>0.333803</td>
      <td>0.860425</td>
      <td>...</td>
      <td>0.288915</td>
      <td>0.262886</td>
      <td>0.337934</td>
      <td>0.407427</td>
      <td>0.917756</td>
      <td>0.363958</td>
      <td>0.744441</td>
      <td>0.071977</td>
      <td>0.521565</td>
      <td>0.337489</td>
    </tr>
    <tr>
      <td>2020-05-01</td>
      <td>0.071323</td>
      <td>0.428552</td>
      <td>0.867582</td>
      <td>0.226246</td>
      <td>0.165900</td>
      <td>0.354248</td>
      <td>0.353920</td>
      <td>0.158203</td>
      <td>0.306319</td>
      <td>0.669884</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.292773</td>
      <td>0.059302</td>
      <td>0.327238</td>
      <td>0.303400</td>
      <td>0.131297</td>
      <td>0.000000</td>
      <td>0.049250</td>
      <td>0.076471</td>
      <td>0.392121</td>
    </tr>
    <tr>
      <td>2020-05-02</td>
      <td>0.066158</td>
      <td>0.388837</td>
      <td>0.898571</td>
      <td>0.232164</td>
      <td>0.218449</td>
      <td>0.341671</td>
      <td>0.323955</td>
      <td>0.165365</td>
      <td>0.353535</td>
      <td>0.670077</td>
      <td>...</td>
      <td>0.244772</td>
      <td>0.254354</td>
      <td>0.274185</td>
      <td>0.461165</td>
      <td>1.000000</td>
      <td>0.263781</td>
      <td>0.743312</td>
      <td>0.102273</td>
      <td>0.025482</td>
      <td>0.535416</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 64 columns</p>
</div>


### Allow ad-hoc filtering for easier experimentation and comparison


```python
# Create function to enable finer tuned testing
def testing_filtration(dataframe, test_columns, test_countries=target_countries):

    # Filter to subset of countries
    country_search = '|'.join(test_countries)
#     print("Countries included", country_search)
    dataframe_filtered = dataframe.iloc[:, dataframe.columns.str.contains(country_search)]

    # Filter to subset of columns
    column_search = '|'.join(test_columns)
    print("Running with columns containing", column_search)
    dataframe_filtered = dataframe_filtered.iloc[:, dataframe_filtered.columns.str.contains(column_search)]

    return dataframe_filtered
```


```python
test_columns = ['response', 'tests', 'apple', 'google']
covid_filtered = testing_filtration(covid_scaled, test_columns)
covid_filtered.tail(3)
```

    Running with columns containing response|tests|apple|google





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Austria_response</th>
      <th>Belgium_response</th>
      <th>Canada_response</th>
      <th>France_response</th>
      <th>Germany_response</th>
      <th>Italy_response</th>
      <th>Netherlands_response</th>
      <th>Norway_response</th>
      <th>Portugal_response</th>
      <th>Singapore_response</th>
      <th>...</th>
      <th>France_google_mobility</th>
      <th>United_Kingdom_google_mobility</th>
      <th>Italy_google_mobility</th>
      <th>Netherlands_google_mobility</th>
      <th>Norway_google_mobility</th>
      <th>Portugal_google_mobility</th>
      <th>Sweden_google_mobility</th>
      <th>Singapore_google_mobility</th>
      <th>Turkey_google_mobility</th>
      <th>United_States_google_mobility</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2020-04-30</td>
      <td>0.072799</td>
      <td>0.478465</td>
      <td>0.847253</td>
      <td>0.242716</td>
      <td>0.224187</td>
      <td>0.371238</td>
      <td>0.377393</td>
      <td>0.168620</td>
      <td>0.333803</td>
      <td>0.860425</td>
      <td>...</td>
      <td>0.288915</td>
      <td>0.262886</td>
      <td>0.337934</td>
      <td>0.407427</td>
      <td>0.917756</td>
      <td>0.363958</td>
      <td>0.744441</td>
      <td>0.071977</td>
      <td>0.521565</td>
      <td>0.337489</td>
    </tr>
    <tr>
      <td>2020-05-01</td>
      <td>0.071323</td>
      <td>0.428552</td>
      <td>0.867582</td>
      <td>0.226246</td>
      <td>0.165900</td>
      <td>0.354248</td>
      <td>0.353920</td>
      <td>0.158203</td>
      <td>0.306319</td>
      <td>0.669884</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.292773</td>
      <td>0.059302</td>
      <td>0.327238</td>
      <td>0.303400</td>
      <td>0.131297</td>
      <td>0.000000</td>
      <td>0.049250</td>
      <td>0.076471</td>
      <td>0.392121</td>
    </tr>
    <tr>
      <td>2020-05-02</td>
      <td>0.066158</td>
      <td>0.388837</td>
      <td>0.898571</td>
      <td>0.232164</td>
      <td>0.218449</td>
      <td>0.341671</td>
      <td>0.323955</td>
      <td>0.165365</td>
      <td>0.353535</td>
      <td>0.670077</td>
      <td>...</td>
      <td>0.244772</td>
      <td>0.254354</td>
      <td>0.274185</td>
      <td>0.461165</td>
      <td>1.000000</td>
      <td>0.263781</td>
      <td>0.743312</td>
      <td>0.102273</td>
      <td>0.025482</td>
      <td>0.535416</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 64 columns</p>
</div>




```python
# #### Define X and Y train

days_train = 14
days_pred = 1
x_train, y_train = structure_supervised(covid_filtered, 'United_States_response', days_train, days_pred, holdout=True)
```


```python
# Calculate number of features
features = x_train.shape[-1]
print(features)
```

    63


------

## Design baseline model and run first experiment

We baseline our case growth predictions using a SimpleRNN model with 100 nodes. Because the other architectures are variations or improvements of a SimpleRNN, this will allow us to easily assess any performance improvements we might gain from a separate model.


```python
# Define RNN model
def model_rnn(days_train, days_pred, features, summary=True):

    rnn = Sequential()
    rnn.add(SimpleRNN(100, input_shape=(days_train, features), return_sequences=False))
    rnn.add(Dense(days_pred, activation="linear"))

    rnn.compile(optimizer=Adam(learning_rate = 0.0005), loss='mae')

    if summary == True:
        rnn.summary()

    return rnn
```


```python
# Define function to train an inputted model
def fit_model(model, x_train=x_train, y_train=y_train, val_split=0.0, epochs=10, verbose=1, batch_size=1):
    history = model.fit(x_train, y_train, validation_split=val_split,
                        epochs=epochs, verbose=verbose, batch_size=batch_size)
    return history
```


```python
rnn = model_rnn(days_train, days_pred, features)

# Fit Simple RNN to US COVID-19 cases
history_rnn = fit_model(rnn, val_split=0.2)
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    simple_rnn (SimpleRNN)       (None, 100)               16400     
    _________________________________________________________________
    dense_8 (Dense)              (None, 1)                 101       
    =================================================================
    Total params: 16,501
    Trainable params: 16,501
    Non-trainable params: 0
    _________________________________________________________________
    Train on 51 samples, validate on 13 samples
    Epoch 1/10
    51/51 [==============================] - 1s 28ms/sample - loss: 0.2398 - val_loss: 0.1380
    Epoch 2/10
    51/51 [==============================] - 0s 9ms/sample - loss: 0.1977 - val_loss: 0.0798
    Epoch 3/10
    51/51 [==============================] - 0s 10ms/sample - loss: 0.1221 - val_loss: 0.1719
    Epoch 4/10
    51/51 [==============================] - 1s 10ms/sample - loss: 0.1428 - val_loss: 0.0740
    Epoch 5/10
    51/51 [==============================] - 1s 10ms/sample - loss: 0.1079 - val_loss: 0.2618
    Epoch 6/10
    51/51 [==============================] - 1s 10ms/sample - loss: 0.1049 - val_loss: 0.1119
    Epoch 7/10
    51/51 [==============================] - 1s 10ms/sample - loss: 0.1220 - val_loss: 0.0914
    Epoch 8/10
    51/51 [==============================] - 1s 10ms/sample - loss: 0.0927 - val_loss: 0.0746
    Epoch 9/10
    51/51 [==============================] - 0s 10ms/sample - loss: 0.1015 - val_loss: 0.0688
    Epoch 10/10
    51/51 [==============================] - 0s 10ms/sample - loss: 0.0822 - val_loss: 0.0573



```python
# Define function to predict
def predict_model(model, x_test):
    prediction = model.predict(x_test)
#     print("Prediction shape", prediction.shape)
    return prediction
```


```python
# Use baseline model to predict from x_train
rnn_pred = predict_model(rnn, x_train)
```


```python
# Define function to plot single-day prediction
def plot_prediction(prediction, y_train, scale_max, ax=None):

    # Rescale data
    prediction_rescaled = prediction * scale_max
    y_train_rescaled = y_train * scale_max

    # Plot figure
    if ax is not None:
        plot_loc=ax
    else:
        plot_loc=plt
        plt.figure(figsize=(20,10))

    plot_loc.plot(y_train_rescaled, label="Actual")
    plot_loc.plot(prediction_rescaled, ls='--', label="Prediction")

    if ax is None:
        plt.legend(fontsize=15, loc=2)
        plt.title(f"Plotting {response} over time", size=20)
        plt.xlabel("Days since first case", size=15)
```


```python
# Plot single-day prediction
plot_prediction(rnn_pred, y_train, max(covid_ranged['United_States_response']))
```


![png](images/output_131_0.png)



```python
# Define function to allow windowed prediction plotting
# when predicting multiple days into the future
def plot_multiday_predictions(y_train, predictions, scale_max, ax=None):

    # Rescale data
    predictions_rescaled = predictions * scale_max
    y_train_rescaled = y_train * scale_max

    # Plot actual data
    cas_obs = []
    for i in range(len(y_train_rescaled)):
        cas_obs.append(y_train_rescaled[i, 0])

    if ax is not None:
        plot_loc=ax
    else:
        plot_loc=plt
        plt.figure(figsize=(20,10))

    plot_loc.plot(np.arange(0, len(cas_obs)), cas_obs, label = 'Observed')
    for i in range(len(predictions_rescaled)):
        plot_loc.plot(np.arange(i, i+predictions_rescaled.shape[-1]), predictions_rescaled[i], c = 'orange')

    if ax is None:
        plt.legend(['Observed', 'Predicted'], fontsize=15, loc=2)
```


```python
# Run sample multi-day prediction
days_train = 14
days_pred = 5
x_train, y_train = structure_supervised(covid_filtered, 'United_States_response', days_train, days_pred, holdout=True)
features = x_train.shape[-1]
rnn = model_rnn(days_train, days_pred, features)
history_rnn = fit_model(rnn, val_split=0.2)
rnn_pred = predict_model(rnn, x_train)
plot_multiday_predictions(y_train, rnn_pred, max(covid_ranged.United_States_response))
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    simple_rnn_1 (SimpleRNN)     (None, 100)               16400     
    _________________________________________________________________
    dense_9 (Dense)              (None, 5)                 505       
    =================================================================
    Total params: 16,905
    Trainable params: 16,905
    Non-trainable params: 0
    _________________________________________________________________
    Train on 51 samples, validate on 13 samples
    Epoch 1/10
    51/51 [==============================] - 1s 29ms/sample - loss: 0.2786 - val_loss: 0.2422
    Epoch 2/10
    51/51 [==============================] - 0s 10ms/sample - loss: 0.1242 - val_loss: 0.2180
    Epoch 3/10
    51/51 [==============================] - 1s 10ms/sample - loss: 0.1064 - val_loss: 0.1312
    Epoch 4/10
    51/51 [==============================] - 1s 11ms/sample - loss: 0.1137 - val_loss: 0.1161
    Epoch 5/10
    51/51 [==============================] - 1s 11ms/sample - loss: 0.0898 - val_loss: 0.1588
    Epoch 6/10
    51/51 [==============================] - 1s 11ms/sample - loss: 0.0835 - val_loss: 0.1356
    Epoch 7/10
    51/51 [==============================] - 1s 10ms/sample - loss: 0.1033 - val_loss: 0.0880
    Epoch 8/10
    51/51 [==============================] - 1s 10ms/sample - loss: 0.0814 - val_loss: 0.1421
    Epoch 9/10
    51/51 [==============================] - 0s 10ms/sample - loss: 0.0664 - val_loss: 0.1112
    Epoch 10/10
    51/51 [==============================] - 1s 10ms/sample - loss: 0.0847 - val_loss: 0.1201



![png](images/output_133_1.png)


## Define various time series architectures
Since the dataset is fairly small, we can afford to run a lot of experiments. Among RNNs, LSTM is generally most powerful, but is also the slowest to fit. While we try single-layer LSTM networks, we therefore also try single-layer GRU, as these sacrifice some expressiveness for faster training speed, to see whether we truly do sacrifice much power. We also try stacking the LSTM network to see if the additional capacity of the model significantly increases its performance. Finally, we try a variant of stacked LSTM that instead uses a second LSTM layer that sees the data in reverse order; such networks have been found to perform very well on some sequence prediction problems.


```python
# Define LSTM model
def model_lstm(days_train, days_pred, features, summary=True):

    lstm = Sequential()
    lstm.add(LSTM(100, activation='relu', return_sequences=False, input_shape=(days_train, features)))
    lstm.add(Dense(days_pred, activation="linear"))

    lstm.compile(optimizer=Adam(learning_rate = 0.0005), loss='mae')

    if summary == True:
        lstm.summary()

    return lstm
```


```python
# Define Stacked LSTM model
def model_stacked_lstm(days_train, days_pred, features, summary=True):

    stacked_lstm = Sequential()
    stacked_lstm.add(LSTM(100, activation = 'relu', batch_input_shape=(1, days_train, features),
                 return_sequences=True, stateful=False))
    stacked_lstm.add(LSTM(50, activation = 'relu', return_sequences = False, stateful=False))
    stacked_lstm.add(Dense(days_pred))

    stacked_lstm.compile(optimizer=Adam(learning_rate = 0.0005), loss='mae')

    if summary == True:
        stacked_lstm.summary()

    return stacked_lstm
```


```python
# Define Bidirectional LSTM model
def model_lstm_bd(days_train, days_pred, features, summary=True):

    lstm_bd = Sequential()
    lstm_bd.add(Bidirectional(LSTM(100, activation='relu'), input_shape=(days_train, features)))
    lstm_bd.add(Dense(days_pred, activation="linear"))

    lstm_bd.compile(optimizer=Adam(learning_rate = 0.0005), loss='mae')

    if summary == True:
        lstm_bd.summary()

    return lstm_bd
```


```python
# Define GRU model
def model_gru(days_train, days_pred, features, summary=True):

    gru = Sequential()
    gru.add(GRU(100, input_shape=(days_train, features), return_sequences=False))
    gru.add(Dense(days_pred, activation="linear"))

    gru.compile(optimizer=Adam(learning_rate = 0.0005), loss='mae')

    if summary == True:
        gru.summary()

    return gru
```


```python
# Define Bidirectional GRU model
def model_gru_bd(days_train, days_pred, features, summary=True):

    gru_bd = Sequential()
    gru_bd.add(Bidirectional(GRU(100, return_sequences=False), input_shape=(days_train, features)))
    gru_bd.add(Dense(days_pred, activation="linear"))

    gru_bd.compile(optimizer=Adam(learning_rate = 0.0005), loss='mae')

    if summary == True:
        gru_bd.summary()

    return gru_bd
```

## Define function to run variety of experiments


```python
# Define a function to run a complete experiment
def run_experiment(model_func, dataframe, agg_column, country_response, test_columns,
                   dates=['2020-02-15', '2020-05-02'], days_train=14, days_pred=1,
                   val_split=0.0, x_test=x_train, ax=None):

    df = generate_multivariate_df(dataframe, 'date', agg_column, fillna=True)
    df = standardize_country_names(df)
    df = filter_to_target_countries(df)
    df = enrich_dataframe(df)
    # Prepare re-scale logic
    df = filter_range(df, dates[0], dates[1])
    if '100k' in agg_column:
        c = country_response.replace("_response", "")
        c_pop = ecdc_country[ecdc_country.country == c]['population_2018'].iloc[0]
        response_scale = max(df[country_response]) * (c_pop/100000)
    else:
        response_scale = max(df[country_response])
    df = scale(df)
    df = testing_filtration(df, test_columns)
    x_train, y_train = structure_supervised(df, country_response, days_train, days_pred, holdout=True)
    features = x_train.shape[-1]
    model = model_func(days_train, days_pred, features, summary=False)
    history = fit_model(model, x_train, y_train, val_split=val_split, verbose=0)
    predictions = predict_model(model, x_train)
    if days_pred > 1:
        plot_multiday_predictions(y_train, predictions, response_scale, ax=ax)
    else:
        plot_prediction(predictions, y_train, response_scale, ax=ax)
```

-------

# Run experiments

## Which RNN architectures best predict different response variables using all time series?

#### Cumulative Cases


```python
model_funcs = [model_rnn, model_stacked_lstm, model_lstm,
               model_gru, model_lstm_bd, model_gru_bd]
model_names = ["Simple RNN", "Stacked LSTM", "Simple LSTM",
              "Simple GRU", "Bidirectional LSTM", "Bidirectional GRU"]
fig, ax = plt.subplots(3, 2, figsize=(20,15))
pos = 0
for i in range(3):
    for j in range(2):
        run_experiment(model_funcs[pos], ecdc_country, 'cases', 'United_States_response',
                       test_columns,
                       dates=['2020-02-15', '2020-05-02'],
                       days_train=14, days_pred=1,
                       val_split=0.2, x_test=x_train, ax=ax[i,j])
        ax[i,j].set_title(f"{model_names[pos]}", size=15)
        pos+=1
plt.suptitle("Cumulative cases since first infection", fontsize=20, color='#237AB4')
ax[0,0].set_ylabel("Number of COVID-19 cases", fontsize=15, color='#237AB4')
ax[0,0].legend(['Observed', 'Predicted'], fontsize=15, loc=2)
fig.subplots_adjust(top=0.93, hspace=0.25)
plt.show()
```

    Running with columns containing response|tests|apple|google
    Running with columns containing response|tests|apple|google
    Running with columns containing response|tests|apple|google
    Running with columns containing response|tests|apple|google
    Running with columns containing response|tests|apple|google
    Running with columns containing response|tests|apple|google



![png](images/output_146_1.png)


#### New Cases and Deaths Per Day


```python
model_funcs = [model_rnn, model_stacked_lstm, model_lstm,
               model_gru, model_lstm_bd, model_gru_bd]
model_names = ["Simple RNN", "Stacked LSTM", "Simple LSTM",
              "Simple GRU", "Bidirectional LSTM", "Bidirectional GRU"]
fig, ax = plt.subplots(3, 2, figsize=(20,15))
pos = 0
for i in range(3):
    for j in range(2):
        run_experiment(model_funcs[pos], ecdc_country, 'new_cases', 'United_States_response',
                       test_columns,
                       dates=['2020-02-15', '2020-05-02'],
                       days_train=14, days_pred=1,
                       val_split=0.0, x_test=x_train, ax=ax[i,j])
        ax[i,j].set_title(f"{model_names[pos]}", size=15)
        pos+=1
plt.suptitle("New cases per day since first infection", fontsize=20, color='#237AB4')
ax[1,0].set_ylabel("Number of COVID-19 cases", fontsize=15, color='#237AB4')
ax[0,0].legend(['Observed', 'Predicted'], fontsize=15, loc=2)
fig.subplots_adjust(top=0.93, hspace=0.25)
plt.show()
```

    Running with columns containing response|tests|apple|google
    Running with columns containing response|tests|apple|google
    Running with columns containing response|tests|apple|google
    Running with columns containing response|tests|apple|google
    Running with columns containing response|tests|apple|google
    Running with columns containing response|tests|apple|google



![png](images/output_148_1.png)



```python
model_funcs = [model_rnn, model_stacked_lstm, model_lstm,
               model_gru, model_lstm_bd, model_gru_bd]
model_names = ["Simple RNN", "Stacked LSTM", "Simple LSTM",
              "Simple GRU", "Bidirectional LSTM", "Bidirectional GRU"]
fig, ax = plt.subplots(3, 2, figsize=(20,15))
pos = 0
for i in range(3):
    for j in range(2):
        run_experiment(model_funcs[pos], ecdc_country, 'new_deaths', 'United_States_response',
                       test_columns,
                       dates=['2020-02-15', '2020-05-02'],
                       days_train=14, days_pred=1,
                       val_split=0.2, x_test=x_train, ax=ax[i,j])
        ax[i,j].set_title(f"{model_names[pos]}", size=15)
        pos+=1
plt.suptitle("New deaths per day since first infection", fontsize=20, color='#237AB4')
ax[0,0].legend(['Observed', 'Predicted'], fontsize=15, loc=2)
ax[1,0].set_ylabel("Number of COVID-19 cases", fontsize=15, color='#237AB4')
fig.subplots_adjust(top=0.93, hspace=0.25)
plt.show()
```

    Running with columns containing response|tests|apple|google
    Running with columns containing response|tests|apple|google
    Running with columns containing response|tests|apple|google
    Running with columns containing response|tests|apple|google
    Running with columns containing response|tests|apple|google
    Running with columns containing response|tests|apple|google



![png](images/output_149_1.png)


#### Conclusion

When predicting a straight line like cumulative cases, simpler architectures do worse because they are highly variable and add unnecessary jumps during predictions. A Stacked LSTM learns the curve best while GRUs, being the least expressive, is unable to learn the shape of the data. We appear to be losing too much predictive power in exchange for the GRU's performance boost. For more variable curves like cases per day, the simpler models appear to follow the data better for single-day predictions. GRUs provide the nearest fit to the data but they still appear to be too jump to be totally accurate. Stacked or Bidirectional LSTMs appear to best blend finding the overall trend while permitting some response to dramatic up/down spikes. As a result, we will continue to use a Stacked LSTM model for future experiments.

------

# Predict a country's case growth using alternate time series

We next turn to the challenge of exclusively using alternate time series to identify the growth rate of coronavirus in a target country. We train varying models on testing records and mobility data from Apple and Google and use that to predict coronavirus case growth. Given the Stacked LSTM appeared to fit best for both single- and multi-day predictions, we continue that forward.

## Does adding alternate time series improve prediction accuracy?

#### Single day Predictions


```python
test_columns = ['response', 'tests', 'apple', 'google']
fig, ax = plt.subplots(2,2, figsize=(20,10))
pos = 1
for i in range(2):
    for j in range(2):
        run_experiment(model_stacked_lstm, ecdc_country, 'new_cases', 'United_States_response',
                       test_columns[:pos],
                       dates=['2020-02-15', '2020-05-02'],
                       days_train=14, days_pred=1,
                       val_split=0.2, x_test=x_train, ax=ax[i,j])
        ax[i,j].set_title(test_columns[:pos], size=15)
        pos+=1
plt.suptitle("New cases per day since first infection", fontsize=20, color='#237AB4')
ax[0,0].set_ylabel("Number of COVID-19 cases", fontsize=15, color='#237AB4')
ax[0,0].legend(['Observed', 'Predicted'], fontsize=15, loc=2)
fig.subplots_adjust(top=0.93, hspace=0.25)
plt.show()
```

    Running with columns containing response
    Running with columns containing response|tests
    Running with columns containing response|tests|apple
    Running with columns containing response|tests|apple|google



![png](images/output_155_1.png)


#### Multiple days out predictions


```python
test_columns = ['response', 'tests', 'apple', 'google']
fig, ax = plt.subplots(2,2, figsize=(20,10))
pos = 1
for i in range(2):
    for j in range(2):
        run_experiment(model_stacked_lstm, ecdc_country, 'new_cases', 'United_States_response',
                       test_columns[:pos],
                       dates=['2020-02-15', '2020-05-02'],
                       days_train=14, days_pred=5,
                       val_split=0.2, x_test=x_train, ax=ax[i,j])
        ax[i,j].set_title(test_columns[:pos], size=15)
        pos+=1
plt.suptitle("New cases per day predicted 5 days into the future", fontsize=20, color='#237AB4')
ax[0,0].set_ylabel("Number of COVID-19 cases", fontsize=15, color='#237AB4')
ax[0,0].legend(['Observed', 'Predicted'], fontsize=15, loc=2)
fig.subplots_adjust(top=0.93, hspace=0.25)
plt.show()
```

    Running with columns containing response
    Running with columns containing response|tests
    Running with columns containing response|tests|apple
    Running with columns containing response|tests|apple|google



![png](images/output_157_1.png)


#### Conclusion

Adding tests alone predicts a higher than reported new cases per day, perhaps reflecting the growing but still insufficient number of tests being performed in the United States. Adding mobility produces the line that appears the best fit to the reported data and while we see some change when using Apple vs Apple and Google, both appear more accurate than alternatives. This remains true for predictions 5 days out as well where including all alternate time series narrows the prediction range throughout the curve.

------

# Predict country's caseload with just that country's alternatives

Having tested whether adding our alternate time series will improve predictions for various responses, we now want to try predicting our response with just the alternate time series. As case counts in various countries may be suspect, testing or mobility data can be easier to collect and used to predict true caseload.


```python
ncols = 4
nrows = 4
fig, ax = plt.subplots(nrows, ncols, figsize=(20,15))
pos = 0
for i in range(nrows):
    for j in range(ncols):
        run_experiment(model_stacked_lstm, ecdc_country, 'new_cases', target_countries[pos] + '_response',
                       target_countries[pos:pos+1],
                       dates=['2020-02-15', '2020-05-02'],
                       days_train=14, days_pred=1,
                       val_split=0.2, x_test=x_train, ax=ax[i,j])
        ax[i,j].set_title(target_countries[pos:pos+1], size=15)
        pos+=1
plt.suptitle("New cases per day predicted from testing and mobility", fontsize=20, color='#237AB4')
ax[0,0].set_ylabel("Number of COVID-19 cases", fontsize=15, color='#237AB4')
ax[0,0].legend(['Observed', 'Predicted'], fontsize=10, loc=2)
fig.subplots_adjust(top=0.93, hspace=0.25)
plt.show()
```

    Running with columns containing France
    Running with columns containing Italy
    Running with columns containing Germany
    Running with columns containing Austria
    Running with columns containing United_Kingdom
    Running with columns containing Spain
    Running with columns containing Portugal
    Running with columns containing Turkey
    Running with columns containing Norway
    Running with columns containing Sweden
    Running with columns containing Belgium
    Running with columns containing Netherlands
    Running with columns containing Switzerland
    Running with columns containing Singapore
    Running with columns containing United_States
    Running with columns containing Canada



![png](images/output_161_1.png)



```python
ncols = 4
nrows = 4
fig, ax = plt.subplots(nrows, ncols, figsize=(20,15))
pos = 0
for i in range(nrows):
    for j in range(ncols):
        run_experiment(model_stacked_lstm, ecdc_country, 'cases_per100k', target_countries[pos] + '_response',
                       target_countries[pos:pos+1],
                       dates=['2020-02-15', '2020-05-02'],
                       days_train=14, days_pred=1,
                       val_split=0.2, x_test=x_train, ax=ax[i,j])
        ax[i,j].set_title(target_countries[pos:pos+1], size=15)
        pos+=1
plt.suptitle("Cumulative cases predicted from testing and mobility", fontsize=20, color='#237AB4')
ax[0,0].set_ylabel("Number of COVID-19 cases", fontsize=15, color='#237AB4')
ax[0,0].legend(['Observed', 'Predicted'], fontsize=10, loc=2)
fig.subplots_adjust(top=0.93, hspace=0.25)
plt.show()
```

    Running with columns containing France
    Running with columns containing Italy
    Running with columns containing Germany
    Running with columns containing Austria
    Running with columns containing United_Kingdom
    Running with columns containing Spain
    Running with columns containing Portugal
    Running with columns containing Turkey
    Running with columns containing Norway
    Running with columns containing Sweden
    Running with columns containing Belgium
    Running with columns containing Netherlands
    Running with columns containing Switzerland
    Running with columns containing Singapore
    Running with columns containing United_States
    Running with columns containing Canada



![png](images/output_162_1.png)



```python
ncols = 4
nrows = 4
fig, ax = plt.subplots(nrows, ncols, figsize=(20,15))
pos = 0
for i in range(nrows):
    for j in range(ncols):
        run_experiment(model_stacked_lstm, ecdc_country, 'new_deaths', target_countries[pos] + '_response',
                       target_countries[pos:pos+1],
                       dates=['2020-02-15', '2020-05-02'],
                       days_train=14, days_pred=1,
                       val_split=0.2, x_test=x_train, ax=ax[i,j])
        ax[i,j].set_title(target_countries[pos:pos+1], size=15)
        pos+=1
plt.suptitle("New deaths per day predicted from testing and mobility", fontsize=20, color='#237AB4')
ax[0,0].set_ylabel("Number of COVID-19 cases", fontsize=15, color='#237AB4')
ax[0,0].legend(['Observed', 'Predicted'], fontsize=10, loc=2)
fig.subplots_adjust(top=0.93, hspace=0.25)
plt.show()
```

    Running with columns containing France
    Running with columns containing Italy
    Running with columns containing Germany
    Running with columns containing Austria
    Running with columns containing United_Kingdom
    Running with columns containing Spain
    Running with columns containing Portugal
    Running with columns containing Turkey
    Running with columns containing Norway
    Running with columns containing Sweden
    Running with columns containing Belgium
    Running with columns containing Netherlands
    Running with columns containing Switzerland
    Running with columns containing Singapore
    Running with columns containing United_States
    Running with columns containing Canada



![png](images/output_163_1.png)


#### Conclusion

We see varied results. Using just testing and mobility rates, we are able to successfully predict new case rates for countries like Norway, Spain, Italy, and Austria. There are a set of countries where the data suggests that new cases should be continuing to increase dramatically beyond what is officially reported. This includes Turkey, the Netherlands, Sweden and Singapore. These countries have generally restricted travel less than their counterparts or already begun to re-open so the model could be using mobility to assess a coming increase that may or may not actually arrive if these countries are re-opening because they've successfully contained the virus.

-----

# How far out into the future can we predict?


```python
days_out = np.linspace(1,28,10).astype(int)

fig, ax = plt.subplots(5,2, figsize=(20,20))
pos = 0
for i in range(5):
    for j in range(2):
        run_experiment(model_stacked_lstm, ecdc_country, 'new_cases_smooth', 'United_States_response',
                       test_columns,
                       dates=['2020-02-15', '2020-05-02'],
                       days_train=14, days_pred=days_out[pos],
                       val_split=0.2, x_test=x_train, ax=ax[i,j])
        ax[i,j].set_title(f"Predicting {days_out[pos]} days out", size=15)
        pos+=1
plt.suptitle("New cases per day", fontsize=20, color='#237AB4')
ax[0,0].set_ylabel("Number of COVID-19 cases", fontsize=15, color='#237AB4')
ax[0,0].legend(['Observed', 'Predicted'], fontsize=15, loc=2)
fig.subplots_adjust(top=0.93, hspace=0.25)
plt.show()
```

    Running with columns containing response|tests|apple|google
    Running with columns containing response|tests|apple|google
    Running with columns containing response|tests|apple|google
    Running with columns containing response|tests|apple|google
    Running with columns containing response|tests|apple|google
    Running with columns containing response|tests|apple|google
    Running with columns containing response|tests|apple|google
    Running with columns containing response|tests|apple|google
    Running with columns containing response|tests|apple|google
    Running with columns containing response|tests|apple|google



![png](images/output_167_1.png)


#### Conclusion
For fairly close prediction windows, the model performs adequately. However, as the time horizon for which we are trying to predict gets longer and longer, the model makes increasingly erratic predictions. This is not altogether unreasonable as the model is fit on sequences of 14 days of history, so for instance predicting 28 days into the future is a tall task. Furthermore, quite a few countries have limited histories of presence in the virus, so the longer our prediction sequences get, the less data the model has to fit on. As such, longer prediction windows means both a harder problem and less data, so it is not unexpected to see it struggle with predictions horizons several times longer than the input sequence.

The model must also strike a balance between accurately predicting the overall trend (otherwise the loss function will penalise it heavily as extrapolating further out makes the squared error exponentially greater) and adequately capturing the local perturbations (the erratic new cases curve means a sudden spike will add considerable error if it is missed by the model). The longer the prediction horizon, the more acute this trade-off becomes.

Consequently, with the data available, our model is most reliable when predicting between 1 and 7 days out. Going much farther out than that, the prediction vector becomes increasingly erratic. For future work, this could likely be improved by smoothing the raw time series by more days to even out some of the more extreme fluctuations that throw the model off keel.

------

------

# Reservoir Computing for COVID-19 Prediction

We saw using traditional RNNs that we were limited to predictions within a week or so of the date. Early in a fast moving pandemic, this warning would be critical for public health officials but once underway, the discussion will turn to what case load will be a month or more. This is critical as countries build plans to re-open their economies and permit people to travel again. To help with longer term predictions, we apply the novel technique of Reservoir Computing, an echo-state recurrent neural network, to pandemic spread prediction. We test two distinct hypotheses.

First, we test the accuracy of short-term RC predictions for the infected population in the days following a prediction using held-out days from the current COVID-19 pandemic. Short-term RCs are quick and computationally cheaper to train than alternatives so can reduce re-training load for epidemiologists if accurate.

Second, we incorporate the concept of observers to test the accuracy of longer-term RC predictions on the order of weeks to months. Because of the structure of viral spread, some countries will benefit from advanced warning by observing accurately collected data from preceding nations. North America is in such a position now with its fight to contain COVID-19. The initial viral outbreak occurred in China at least two months before significant spread was reported in the state of Washington in the United States. As our Multi-Model showed, the United States is an outlier from the remainder of the world, meaning its experience can not be accurately predicted by comparing it to other countries. Instead, we predict case growth in Canada, a country with a similar level of warning but a public health response more in line with other nations. We select a handful of countries with varying sizes and durations of curves, all of which were experienced before Canada began to experience an increased case load.


```python
# Save dataframe specific to the United States
usa = ecdc_country[ecdc_country['country'] == 'United_States']
usa.tail(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>day</th>
      <th>month</th>
      <th>year</th>
      <th>new_cases</th>
      <th>new_deaths</th>
      <th>country</th>
      <th>geoID</th>
      <th>country_code</th>
      <th>population_2018</th>
      <th>...</th>
      <th>cases</th>
      <th>deaths</th>
      <th>cases_per100k</th>
      <th>deaths_per100k</th>
      <th>new_cases_per100k</th>
      <th>new_deaths_per100k</th>
      <th>new_cases_smooth</th>
      <th>new_cases_per100k_smooth</th>
      <th>day_zero</th>
      <th>days_since_zero</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>15491</td>
      <td>2020-05-06</td>
      <td>6</td>
      <td>5</td>
      <td>2020</td>
      <td>23841</td>
      <td>2144</td>
      <td>United_States</td>
      <td>US</td>
      <td>USA</td>
      <td>327167434.0</td>
      <td>...</td>
      <td>1204475</td>
      <td>71078</td>
      <td>368.152473</td>
      <td>21.725267</td>
      <td>7.287094</td>
      <td>0.655322</td>
      <td>26929.8</td>
      <td>8.231198</td>
      <td>2020-01-21</td>
      <td>106</td>
    </tr>
    <tr>
      <td>15490</td>
      <td>2020-05-07</td>
      <td>7</td>
      <td>5</td>
      <td>2020</td>
      <td>24128</td>
      <td>2353</td>
      <td>United_States</td>
      <td>US</td>
      <td>USA</td>
      <td>327167434.0</td>
      <td>...</td>
      <td>1228603</td>
      <td>73431</td>
      <td>375.527290</td>
      <td>22.444471</td>
      <td>7.374817</td>
      <td>0.719204</td>
      <td>24964.4</td>
      <td>7.630466</td>
      <td>2020-01-21</td>
      <td>107</td>
    </tr>
    <tr>
      <td>15489</td>
      <td>2020-05-08</td>
      <td>8</td>
      <td>5</td>
      <td>2020</td>
      <td>28369</td>
      <td>2239</td>
      <td>United_States</td>
      <td>US</td>
      <td>USA</td>
      <td>327167434.0</td>
      <td>...</td>
      <td>1256972</td>
      <td>75670</td>
      <td>384.198386</td>
      <td>23.128830</td>
      <td>8.671095</td>
      <td>0.684359</td>
      <td>24780.6</td>
      <td>7.574287</td>
      <td>2020-01-21</td>
      <td>108</td>
    </tr>
    <tr>
      <td>15488</td>
      <td>2020-05-09</td>
      <td>9</td>
      <td>5</td>
      <td>2020</td>
      <td>26957</td>
      <td>1510</td>
      <td>United_States</td>
      <td>US</td>
      <td>USA</td>
      <td>327167434.0</td>
      <td>...</td>
      <td>1283929</td>
      <td>77180</td>
      <td>392.437898</td>
      <td>23.590367</td>
      <td>8.239512</td>
      <td>0.461537</td>
      <td>25177.6</td>
      <td>7.695631</td>
      <td>2020-01-21</td>
      <td>109</td>
    </tr>
    <tr>
      <td>15487</td>
      <td>2020-05-10</td>
      <td>10</td>
      <td>5</td>
      <td>2020</td>
      <td>25612</td>
      <td>1614</td>
      <td>United_States</td>
      <td>US</td>
      <td>USA</td>
      <td>327167434.0</td>
      <td>...</td>
      <td>1309541</td>
      <td>78794</td>
      <td>400.266305</td>
      <td>24.083693</td>
      <td>7.828408</td>
      <td>0.493325</td>
      <td>25781.4</td>
      <td>7.880185</td>
      <td>2020-01-21</td>
      <td>110</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
# Plot USA Cases in Linear and Log Scale
fig, axs = plt.subplots(1,2, figsize=(20,8))

axs[0].set_yscale('linear')
axs[1].set_yscale('log')
axs[0].set_title("Linear Scale", size=20)
axs[1].set_title("Log Scale", size=20)

for i in range(2):
    axs[i].bar(usa['date'][75:], usa['cases'][75:], color='blue', label="Cases")
    axs[i].bar(usa['date'][75:], usa['deaths'][75:], color='red', label="Deaths")

    axs[i].yaxis.grid(color='gray', linestyle='dashed')
    axs[i].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    axs[i].tick_params(axis='x', labelrotation=45)

axs[0].legend(fontsize=15)
plt.show()
```


![png](images/output_174_0.png)


## Short-Term RC Predictions of Infection Rate


```python
# Select starting response column for design
response_column = 'new_cases_smooth'
```


```python
# Filter to data necessary for RC
usa_rc = usa[usa[response_column] > 100][response_column].reset_index(drop=True)
usa_rc
```




    0       119.0
    1       158.4
    2       194.8
    3       246.0
    4       324.0
           ...   
    57    26929.8
    58    24964.4
    59    24780.6
    60    25177.6
    61    25781.4
    Name: new_cases_smooth, Length: 62, dtype: float64




```python
# Define prepare_data function
def prepareData(target, train_perc=0.9, plotshow=False):
    datalen =  len(target)        
    trainlen = int(train_perc*datalen)
    testlen  = datalen-trainlen

# Train/Test sets
    trainTarget = np.array(target[:trainlen])
    testTarget  = np.array(target[trainlen:trainlen+testlen]    )
    inputTrain = np.ones(trainlen)
    inputTest  = np.ones(testlen)

    if plotshow:
        plt.figure(figsize=(14,3))
        plt.plot(range(0,trainlen), trainTarget,'g',label='Train')
        plt.plot(range(trainlen,trainlen+testlen), testTarget,'-r',label='Test')
        plt.legend(loc=(0.1,1.1),fontsize=18,ncol=2)
        plt.tight_layout()
        plt.show()

    return trainTarget, testTarget, inputTrain, inputTest
```


```python
# Prepare US data
trainTarget, testTarget, inputTrain, inputTest = prepareData(usa_rc, train_perc=0.90, plotshow=True)
```


![png](images/output_179_0.png)



```python
# Plot shapes
trainTarget.shape, testTarget.shape, inputTrain.shape, inputTest.shape
```




    ((55,), (7,), (55,), (7,))




```python
# Define constant hyper-parameters
n_outputs = 1
n_reservoir = 1000
noise = 0.0001
random_state=42
```


```python
### Make manual prediction

# Define ESN
esn_st = ESN(n_inputs = 1,
          spectral_radius = 1.5,
          sparsity = 0.5,
          n_outputs=n_outputs,
          n_reservoir=n_reservoir,
          noise=noise,
          random_state=random_state)

# Train ESN
pred_training = esn_st.fit(inputTrain, trainTarget)
```


```python
# Define MSE function
def MSE(prediction,target):
    return np.sqrt(np.mean((prediction.flatten() - target.flatten() )**2))
```


```python
# Make quick prediction and return MSE
prediction = esn_st.predict(inputTest)
print("Quick Prediction MSE:", round(MSE(prediction, testTarget), 5))
```

    Quick Prediction MSE: 2804.91109



```python
def residuals(prediction,target):
    return (target.flatten() - prediction.flatten())
```


```python
# Plot training, predictions and ground truth
fig, axs = plt.subplots(2,1, figsize=(20,10))
len_train = len(trainTarget)
len_test = len(testTarget)
axs[0].plot(range(0,len_train), trainTarget, label="Training Data")
axs[0].plot(range(len_train, len_train+len_test), testTarget, label="Test Data")
axs[0].plot(range(len_train, len_train+len_test), prediction, label="Predictions")
res = residuals(prediction, testTarget)
axs[0].plot(range(len_train, len_train+len_test), res, ls="--", label="Residuals")

axs[1].plot(range(len_train, len_train+len_test), res, ls="--", color="red", label="Residuals")

axs[0].set_title("Short-Term Forecast", size=20)
axs[1].set_title("Zoomed to Residuals", size=20)
axs[0].legend(loc=2, fontsize=10)
axs[1].legend(loc=2, fontsize=10)
plt.show()
```


![png](images/output_186_0.png)


#### Vary the Response

We next define a function to run a number of experiments to see how Short-Term RC predicts different important response variables.


```python
# Define function to quickly run experiment
def run_rc_experiment(response, data, model, train_pct=0.9, min_value=False, plotshow=True):
    data_rc = data[data[response] != 0][response].reset_index(drop=True)
    if min_value is not False:
        data_rc = data[data[response] >= min_value][response].reset_index(drop=True)
    trainTarget, testTarget, inputTrain, inputTest = prepareData(data_rc, train_perc=train_pct, plotshow=plotshow)
    pred_training = model.fit(inputTrain, trainTarget)
    prediction = model.predict(inputTest)
    print("Quick Prediction MSE:", round(MSE(prediction, testTarget), 5))

    # Plot training, predictions and ground truth
    fig, axs = plt.subplots(2,1, figsize=(20,10))
    len_train = len(trainTarget)
    len_test = len(testTarget)
    axs[0].plot(range(0,len_train), trainTarget, label="Training Data")
    axs[0].plot(range(len_train, len_train+len_test), testTarget, label="Test Data")
    axs[0].plot(range(len_train, len_train+len_test), prediction, label="Predictions")
    res = residuals(prediction, testTarget)
    axs[0].plot(range(len_train, len_train+len_test), res, ls="--", label="Residuals")

    axs[1].plot(range(len_train, len_train+len_test), res, ls="--", color="red", label="Residuals")

    axs[0].set_title("Short-Term Forecast", size=20)
    axs[1].set_title("Zoomed to Residuals", size=20)
    axs[0].legend(loc=2, fontsize=10)
    axs[1].legend(loc=2, fontsize=10)
    plt.show()
```


```python
run_rc_experiment('cases', usa, esn_st, train_pct=0.90, min_value=100, plotshow=False)
```

    Quick Prediction MSE: 845046.68581



![png](images/output_189_1.png)



```python
run_rc_experiment('cases_per100k', usa, esn_st, train_pct=0.90, min_value=10, plotshow=False)
```

    Quick Prediction MSE: 10.72721



![png](images/output_190_1.png)



```python
run_rc_experiment('deaths', usa, esn_st, train_pct=0.90, min_value=100, plotshow=False)
```

    Quick Prediction MSE: 39498.30367



![png](images/output_191_1.png)



```python
run_rc_experiment('deaths_per100k', usa, esn_st, train_pct=0.9, min_value=False, plotshow=False)
```

    Quick Prediction MSE: 1.51474



![png](images/output_192_1.png)



```python
run_rc_experiment('new_cases', usa, esn_st, train_pct=0.90, min_value=100, plotshow=False)
```

    Quick Prediction MSE: 3473.01395



![png](images/output_193_1.png)



```python
run_rc_experiment('new_cases_per100k', usa, esn_st, train_pct=0.90, min_value=False, plotshow=False)
```

    Quick Prediction MSE: 94.76036



![png](images/output_194_1.png)



```python
run_rc_experiment('new_cases_smooth', usa, esn_st, train_pct=0.90, min_value=1000, plotshow=False)
```

    Quick Prediction MSE: 3695.16775



![png](images/output_195_1.png)



```python
run_rc_experiment('new_cases_per100k_smooth', usa, esn_st, train_pct=0.90, min_value=1, plotshow=False)
```

    Quick Prediction MSE: 290.00796



![png](images/output_196_1.png)


#### Conclusion

Over the short term, several days out, RC performs well when predicting either variable responses like new cases or new deaths or standardized responses like cases_per100k. It does very poorly on absolute cases and standardized variable responses like new_cases_per100k. Accuracy is significantly improved by filtering time series to values reflecting the start of exponential growth (i.e., the first bend in the curve). By limiting to learning on time series only once the increase has begun in earnest, the RC does not over learn the long starting tail of low case counts and prematurely predict a decline (or return to mean of zero).

-----

## Long-Term RC Predictions

We next move to predictions farther than a few days out. For our observers, we identified a mix of Asian and European countries with varying outcomes. Singapore and South Korea maintained very low case loads throughout their curve though they hit their peak at different times in their infection cycle. Italy and France were among some of the hardest hit countries in the world, hitting their peaks at different times as well. Canada has seen a higher case count than our Asian nation examples but slower overall growth than our European nations.


```python
# Define hyperparameters
min_value = 25
response_column = 'new_cases_smooth'
```


```python
# Save dataframe specific to the Canada
canada = ecdc_country[ecdc_country['country'] == 'Canada']

canada.tail(14)

# Filter to data necessary for RC
canada_rc = canada[canada[response_column] != 0][response_column].reset_index(drop=True)

# Filter to values only above certain size
canada_rc = canada_rc[canada_rc > min_value]

# Save dataframe specific to the France
france = ecdc_country[ecdc_country['country'] == 'France']
france.tail(14)

# Filter to data necessary for RC
france_rc = france[france[response_column] != 0][response_column].reset_index(drop=True)

# Filter to values only above certain size
france_rc = france_rc[france_rc > min_value]

# Save dataframe specific to Singapore
italy = ecdc_country[ecdc_country['country'] == 'Italy']
italy.tail(14)

# Filter to data necessary for RC
italy_rc = italy[italy[response_column] != 0][response_column].reset_index(drop=True)

# Filter to values only above certain size
italy_rc = italy_rc[italy_rc > min_value]

# Save dataframe specific to Singapore
singapore = ecdc_country[ecdc_country['country'] == 'Singapore']
singapore.tail(14)

# Filter to data necessary for RC
singapore_rc = singapore[singapore[response_column] != 0][response_column].reset_index(drop=True)

# Filter to values only above certain size
singapore_rc = singapore_rc[singapore_rc > 10]

# Save dataframe specific to South Korea
korea = ecdc_country[ecdc_country['country'] == 'South_Korea']
korea.tail(14)

# Filter to data necessary for RC
korea_rc = korea[korea[response_column] != 0][response_column].reset_index(drop=True)

# Filter to values only above certain size
korea_rc = korea_rc[korea_rc > min_value]

# Save dataframe specific to South Korea
japan = ecdc_country[ecdc_country['country'] == 'Japan']
japan.tail(14)

# Filter to data necessary for RC
japan_rc = japan[japan[response_column] != 0][response_column].reset_index(drop=True)

# Filter to values only above certain size
japan_rc = japan_rc[japan_rc > min_value]
```


```python
# Plot four chosen countries data
plt.figure(figsize=(20,5))
plt.plot(canada_rc, label="Canada")
plt.plot(france_rc, label="France")
plt.plot(italy_rc, label="Italy")
plt.plot(singapore_rc, label="Singapore")
plt.plot(korea_rc, label="Korea")
plt.plot(japan_rc, label="Japan")


plt.legend(fontsize=15)
plt.show()
```


![png](images/output_202_0.png)



```python
### Prepare the data

trainTarget, testTarget, inputTrain, inputTest = prepareData(canada_rc, train_perc=0.5, plotshow=True)
```


![png](images/output_203_0.png)



```python
### Prepare observers
print("Before stack:", trainTarget.shape, testTarget.shape, inputTrain.shape, inputTest.shape)

# Calculate lengths
trainlen = len(inputTrain)
testlen = len(inputTest)
print(trainlen, testlen)

# Split Canada
canada_train = canada_rc[:trainlen]
canada_test = canada_rc[trainlen:]

# Split Singapore
sg_train = singapore_rc[:trainlen]
sg_test = singapore_rc[trainlen:]

# Split Italy
italy_train = italy_rc[:trainlen]
italy_test = italy_rc[trainlen:]

# Split France
france_train = france_rc[:trainlen]
france_test = france_rc[trainlen:]

# Split Korea
korea_train = korea_rc[:trainlen]
korea_test = korea_rc[trainlen:]

# Split Japan
japan_train = japan_rc[:trainlen]
japan_test = japan_rc[trainlen:]

# Print shapes
print("Train", inputTrain.shape, canada_train.shape, japan_train.shape, korea_train.shape, sg_train.shape, italy_train.shape, france_train.shape)
print("Test", inputTest.shape, canada_test.shape, japan_test.shape, korea_test.shape, sg_test.shape, italy_test.shape, france_test.shape)

# Predict minimum of test set
testlen_min = min(inputTest.shape, canada_test.shape, japan_test.shape, korea_test.shape, sg_test.shape, france_test.shape, italy_test.shape)[0]
inputTest = inputTest[:testlen_min]
testTarget = testTarget[:testlen_min]
canada_test = canada_test[:testlen_min]
italy_test = italy_test[:testlen_min]
france_test = france_test[:testlen_min]
sg_test = sg_test[:testlen_min]
korea_test = korea_test[:testlen_min]
japan_test = japan_test[:testlen_min]

print("Test After Update", inputTest.shape, canada_test.shape, japan_test.shape, korea_test.shape, sg_test.shape, italy_test.shape, france_test.shape)

# Stack input and observers
inputTrain = np.stack((inputTrain, japan_train, korea_train, sg_train, italy_train, france_train), axis=1)
inputTest = np.stack((inputTest, japan_test, korea_test, sg_test, italy_test, france_test), axis=1)
print("After stack:", trainTarget.shape, testTarget.shape, inputTrain.shape, inputTest.shape)
```

    Before stack: (28,) (29,) (28,) (29,)
    28 29
    Train (28,) (28,) (28,) (28,) (28,) (28,) (28,)
    Test (29,) (29,) (37,) (28,) (29,) (49,) (41,)
    Test After Update (28,) (28,) (28,) (28,) (28,) (28,) (28,)
    After stack: (28,) (28,) (28, 6) (28, 6)



```python
### Make search grid for hyper-parameters spectral-radius and sparsity

# Define manual hyper-parameter tests
spectral_radius = np.linspace(1, 9, 10)
sparsity = np.linspace(0.1, 0.9, 10)

test_mse = {}

# Grid search across two hyper-parameters
for sr in spectral_radius:
    for sp in sparsity:
        # Define ESN
        esn_lt = ESN(n_inputs = 6,
                  spectral_radius = sr,
                  sparsity = sp,
                  n_outputs=n_outputs,
                  n_reservoir=n_reservoir,
                  noise=noise,
                  random_state=random_state)

        # Train ESN
        pred_training = esn_lt.fit(inputTrain, trainTarget)

        # Make quick prediction and return MSE
        prediction = esn_lt.predict(inputTest)
        mse = MSE(prediction, testTarget)
        params = [sr, sp]
        test_mse[mse] = params
        #print(f"SR: {sr}, SP: {sp} == MSE: {round(mse, 5)}")
```


```python
min_mse = min(test_mse.keys())
print("Lowest testing MSE:", min_mse)
```

    Lowest testing MSE: 258.0286841160495



```python
# Define optimal hyper-parameters
sr_best = test_mse[min_mse][0]
sp_best = test_mse[min_mse][1]
print("SR: ", str(sr_best), ", SP: ", str(sp_best))
```

    SR:  5.444444444444445 , SP:  0.6333333333333333



```python
# Make a quick prediction
esn_lt = ESN(n_inputs = 6,
                  spectral_radius = sr_best,
                  sparsity = sp_best,
                  n_outputs=n_outputs,
                  n_reservoir=n_reservoir,
                  noise=noise,
                  random_state=random_state)

# Train ESN
pred_training = esn_lt.fit(inputTrain, trainTarget)

# Make quick prediction and return MSE
prediction = esn_lt.predict(inputTest)
mse = MSE(prediction, testTarget)
print("Quick Prediction MSE:", mse)
```

    Quick Prediction MSE: 258.0286841160495



```python
# Plot training, predictions and ground truth
fig, axs = plt.subplots(2,1, figsize=(20,10))
len_train = len(trainTarget)
len_test = len(testTarget)
axs[0].plot(range(0,len_train), trainTarget, label="Training Data")
axs[0].plot(range(len_train, len_train+len_test), testTarget, label="Test Data")
axs[0].plot(range(len_train, len_train+len_test), prediction, label="Predictions")
res = residuals(prediction, testTarget)
axs[0].plot(range(len_train, len_train+len_test), res, ls="--", label="Residuals")

axs[1].plot(range(len_train, len_train+len_test), res, ls="--", color="red", label="Residuals")
axs[1].plot(range(len_train, len_train+len_test), np.repeat(np.mean(res), len_test), ls="-", color="red", alpha=0.5, label="Mean Residual")

axs[0].set_title("Long-Term Forecast", size=20)
axs[1].set_title("Zoomed to Residuals", size=20)
axs[0].legend(loc=2, fontsize=10)
axs[1].legend(loc=2, fontsize=10)
plt.show()
```


![png](images/output_209_0.png)


#### Conclusion

Using observers with a variety of experiences, we were able to predict Canada's expected rate of new cases (smoothed) fairly well 28 days ahead of time. We set a minimum number of new cases at 25 to eliminate the long tail that might artificially send predictions towards zero too early. The forecast has a mean residual of almost exactly zero but can provide predictions as incorrect as +/- 400 new cases per day. Furthermore, even though the mean residual in our test set is $\sim 0$, the value of the residual seems to be a function of time, starting highly negative and growing with time.

That said, this is impressive performance. The fact is the model predicts as far into the future as it sees during training, and does an impressive job at it.

-----

-----

# Predicting what lies ahead
Having evaluated all these models on the various problem structures, we turn towards what the future has in store for the United States.


```python
# Re-build data structures necessary for predictions
x_train, y_train = structure_supervised(covid_filtered, 'United_States_response', 14, 7, holdout=True)
rc_usa = usa[usa['new_cases_smooth'] > 100]['new_cases_smooth'].reset_index(drop=True)
case_scale = max(usa['new_cases_smooth'])
print(case_scale)
```

    32785.0


### Designed Neural Network


```python
# Calculate numbers for re-scaling
mul_predictions = mul.predict(usa_x)[-1]
usa_pop = growth_curve_all[growth_curve_all.country == 'United_States']['population_2018'].iloc[0]
max_count = max(growth_curve_all['new_cases_per100k_smooth'])
mul_predictions = mul_predictions * max_count * (usa_pop/100000)
mul_predictions
```




    array([15912.745, 15369.569, 14476.506, 14135.635, 13915.881, 13831.641,
           13033.562], dtype=float32)



### Stacked LSTM


```python
# Re-define Stacked LSTM
stacked_lstm = model_stacked_lstm(14, 7, features)

# Fit model to US COVID-19 cases
history_stacked_lstm = fit_model(stacked_lstm, val_split=0.2)
```

    Model: "sequential_86"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm_148 (LSTM)              (1, 14, 100)              65600     
    _________________________________________________________________
    lstm_149 (LSTM)              (1, 50)                   30200     
    _________________________________________________________________
    dense_94 (Dense)             (1, 7)                    357       
    =================================================================
    Total params: 96,157
    Trainable params: 96,157
    Non-trainable params: 0
    _________________________________________________________________
    Train on 51 samples, validate on 13 samples
    Epoch 1/10
    51/51 [==============================] - 4s 74ms/sample - loss: 0.2839 - val_loss: 0.9542
    Epoch 2/10
    51/51 [==============================] - 1s 28ms/sample - loss: 0.1463 - val_loss: 0.1352
    Epoch 3/10
    51/51 [==============================] - 1s 28ms/sample - loss: 0.0366 - val_loss: 0.0811
    Epoch 4/10
    51/51 [==============================] - 1s 28ms/sample - loss: 0.0328 - val_loss: 0.0810
    Epoch 5/10
    51/51 [==============================] - 1s 29ms/sample - loss: 0.0395 - val_loss: 0.0818
    Epoch 6/10
    51/51 [==============================] - 1s 28ms/sample - loss: 0.0338 - val_loss: 0.0893
    Epoch 7/10
    51/51 [==============================] - 2s 35ms/sample - loss: 0.0295 - val_loss: 0.0501
    Epoch 8/10
    51/51 [==============================] - 1s 29ms/sample - loss: 0.0272 - val_loss: 0.0724
    Epoch 9/10
    51/51 [==============================] - 1s 28ms/sample - loss: 0.0282 - val_loss: 0.1765
    Epoch 10/10
    51/51 [==============================] - 1s 28ms/sample - loss: 0.0279 - val_loss: 0.0629



```python
stacked_lstm_predictions = stacked_lstm.predict(x_train)[-1]*case_scale
stacked_lstm_predictions
```




    array([30657.447, 31499.81 , 26028.436, 28344.29 , 29460.48 , 27571.352,
           31419.28 ], dtype=float32)



### Short-Term RC


```python
### Make manual prediction

# Prepare US data
trainTarget, testTarget, inputTrain, inputTest = prepareData(usa_rc, train_perc=1.0, plotshow=False)

# Train ESN
pred_usa = esn_st.fit(inputTrain, trainTarget)

# Make prediction
esn_st_prediction = esn_st.predict(inputTrain)[-7:]
esn_st_prediction
```




    array([[24527.68386672],
           [22434.61723253],
           [23499.43262978],
           [29172.72802673],
           [24060.40060189],
           [25160.30784369],
           [26760.9091952 ]])



### Plot predictions


```python
plt.figure(figsize=(20,10))
plt.plot(usa['date'][75:], usa['new_cases_smooth'][75:], label = 'Observed')
plt.plot(usa['date'][-7:] + pd.DateOffset(days=7), mul_predictions, ls='--', label="Designed Neural Network")
plt.plot(usa['date'][-7:] + pd.DateOffset(days=7), stacked_lstm_predictions, ls='--', label="Stacked LSTM")
plt.plot(usa['date'][-7:] + pd.DateOffset(days=7), esn_st_prediction, ls='--', label="Short Term Reservoir Computing")
plt.title("New Case Predictions from Today (May 10th, 2020)", size=25)
plt.ylabel("New cases per death (smooth)", size=15)
plt.xlabel("Date", size=15)
plt.legend(loc=2, fontsize=15)
plt.show()
```


![png](images/output_223_0.png)


# Concluding Remarks

As we can see from our final predictions, each model interacts with the developing COVID-19 data in slightly distinctive ways. The Stacked LSTM model maintains the longest "memory" and so is likeliest to expect a return to traditional growth expectations seen in the first part of the time series. The Short-Term RC model is most frequently applied to chaotic systems so it tends to expect a continued up-and-down from new case growth. Our self-designed neural network was trained on the growth curves of all countries so it has learned from other countries that the general trend is to decrease after hitting a peak. The divergence between these three model predictions suggests that the United States remains on a plateau in its fight to contain COVID-19. We are no longer seeing the high earlier growth but neither are we seeing the decline that other nations are. As new data comes in, our models can be updated to get a better sense of which direction the case load will go. If they begin to agree, we can be more confident that case load is migrating in that direction. Until then, the fight against COVID-19 requires continued public health expertise and improved policymaking. With better public policy, our models will begin to reflect the decline we all want to see.
