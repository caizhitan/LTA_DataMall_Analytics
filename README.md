# LTA_DataMall_Analytics

This project is writen in Python using libaries such as Pandas, Numpy, Matplotlib for Data Analytics.

## This documentation will be split into 2 sections:

### Data Processing

- Collecting our Dataset.
- Understanding our Dataset.
- Cleaning our Dataset.
- Transforming our Dataset.

### Data Analysis

- Performing analysis.
- Visualise with graphs.
- Share findings and summary.

# Data Processing

In this section I will explain my thought process of processing the data for analysis.

## Collecting our Dataset.

### Passenger Volume By Train Stations (Dataset #1)

![Dataset #1](https://github.com/caizhitan/LTA_DataMall_Analytics/assets/150103035/79d3914c-744f-4438-9f43-a85ffa311803)

### Passenger Volume By Origin Destination Train Station (Dataset #2)

![Dataset #2](https://github.com/caizhitan/LTA_DataMall_Analytics/assets/150103035/b14c832b-4416-426d-84b1-38fa8a8f574e)

The documentation states a brief description of what the Dataset contains and its Update Frequency.
Annex A & Annex B gives us more detailed explaination of our variables in the dataset.

We can retrieve our dataset by date by sending a Request Qurey Parameter: `Date=YYYYMM`
This will give us a temporarily AWS S3 Bucket link for downloading.

![Postman API](https://github.com/caizhitan/LTA_DataMall_Analytics/assets/150103035/ef80df6f-b09d-4019-bf9e-1586f33e76c2)


## Understanding our Dataset.

### Annex A (Sample of Dataset #1)

| YEAR_MONTH | DAY_TYPE         | TIME_PER_HOUR | PT_TYPE | PT_CODE   | TOTAL_TAP_IN_VOLUME | TOTAL_TAP_OUT_VOLUME |
| ---------- | ---------------- | ------------- | ------- | --------- | ------------------- | -------------------- |
| 2018-05    | WEEKDAY          | 15            | TRAIN   | EW14-NS26 | 56019               | 37614                |
| 2018-05    | WEEKENDS/HOLIDAY | 15            | TRAIN   | EW14-NS26 | 13385               | 10878                |

Referencing from the above Annex A photo provided by the API Documentation we can see that we are given:

1. YEAR_MONTH : YYYY-MM
1. DAY_TYPE : Weekday or Weekends/Holiday
1. TIME*PER_HOUR : Hour of the day \_e.g 15 = 1500hrs - 1559hrs*
1. PT_TYPE : Transportation Type
1. TOTAL_TAP_IN_VOLUME : Number of Tap Ins at Station during hour of the day
1. TOTAL_TAP_OUT_VOLUME : Number of Tap Outs at Station during hour of the day

### Annex B (Sample of Dataset #2)

| YEAR_MONTH | DAY_TYPE         | TIME_PER_HOUR | PT_TYPE | ORIGIN_PT_CODE | DESTINATION_PT_CODE | TOTAL_TRIPS |
| ---------- | ---------------- | ------------- | ------- | -------------- | ------------------- | ----------- |
| 2018-05    | WEEKDAY          | 17            | TRAIN   | CC28           | CC1-NE6-NS24        | 111         |
| 2018-05    | WEEKENDS/HOLIDAY | 17            | TRAIN   | CC28           | CC1-NE6-NS24        | 39          |

1. YEAR_MONTH : YYYY-MM
1. DAY_TYPE : Weekday or Weekends/Holiday
1. TIME*PER_HOUR : Hour of the day \_e.g 17 = 1700hrs - 1759hrs*
1. PT_TYPE : Transportation Type
1. ORIGIN_PT_CODE : Origin Train Station Code
1. DESTINATION_PT_CODE : Destination Train Station Code
1. TOTAL_TRIPS : Number of trips made between Origin & Destination train station during hour of the day

## Cleaning our Dataset #1.

### Writing our Python Code

#### Import required libraries

```Python
import pandas as pd
```

For processing our dataset we will only need pandas library.

#### Finding Blank Columns or Rows

```Python
blanks = dfCombinedFirst.isna()
print(blanks)
```

As expected, our dataset does not contain any blanks.

#### Finding Duplicated Columns or Rows

```Python
duplicates = dfCombinedFirst.duplicated().count
print(duplicates)
```

As expected, our dataset also does not contain any duplicated entries of data.

#### Filtering Unnecessary Columns

```Python
dfCombinedFirst = dfCombinedFirst.rename(columns={'PT_TYPE': 'PT_NAME'})
```

As PT_TYPE is repeatedly "TRAIN" we will rename the column to PT_NAME and use it for Mapping PT_CODE with its respective Station Name.

## Transforming our Dataset.

### Writing our Python Code

#### Mapping Station Codes with Station Name (With Dataset from [Kaggle](https://www.kaggle.com/datasets/cztandata/singapore-train-station-locations))

```Python
# Map Station Names (PT_NAME) with Station Codes (PT_CODE)
dfCombinedFirst['PT_CODE_FirstPart'] = dfCombinedFirst['PT_CODE'].str.split('/').str[0]
csv_df = pd.read_csv('./datasets/TrainStationCodes.csv') # Our Kaggle dataset
code_name_mapping = dict(zip(csv_df['stn_code'], csv_df['mrt_station_english']))
dfCombinedFirst['PT_NAME'] = dfCombinedFirst['PT_CODE_FirstPart'].map(code_name_mapping)
dfCombinedFirst = dfCombinedFirst.drop('PT_CODE_FirstPart', axis=1)

print(dfCombinedFirst)
```

Here we use another csv file to check PT_CODE and replace PT_NAME column from "TRAIN" to the actual English Station Name.

#### Changing Month_Year & Time_Per_Year column to datetime format

```Python
# Convert YEAR_MONTH to the last day of the month
dfCombinedFirst['YEAR_MONTH'] = pd.to_datetime(dfCombinedFirst['YEAR_MONTH']).dt.to_period('M').dt.to_timestamp('M') + pd.offsets.MonthEnd(0)

# Combine with TIME_PER_HOUR to create a full datetime
dfCombinedFirst['DATETIME'] = dfCombinedFirst.apply(lambda row: pd.Timestamp(year=row['YEAR_MONTH'].year,
                                                   month=row['YEAR_MONTH'].month,
                                                   day=row['YEAR_MONTH'].day,
                                                   hour=row['TIME_PER_HOUR']), axis=1)

# Drop the original YEAR_MONTH column
dfCombinedFirst.drop('YEAR_MONTH', axis=1, inplace=True)

# Make DATETIME the first column by creating a new DataFrame with the desired column order
dfCombinedFirst = dfCombinedFirst[['DATETIME'] + [col for col in dfCombinedFirst.columns if col != 'DATETIME']]

# Drop the origin Time_Per_Hour columns
dfCombinedFirst.drop('TIME_PER_HOUR', axis=1, inplace=True)
```
As the data rows are a sum for the specific month, we will set the day to be the last day of the month. For the time, we use the TIME_PER_HOUR Column from the dataset.

#### Perform Label Encoding: (Weekdays to 0) & (Weekends/Holidays to 1)
```Python
dfCombinedFirst['DAY_TYPE'] = dfCombinedFirst['DAY_TYPE'].map({'WEEKDAY': 0, 'WEEKENDS/HOLIDAY': 1})
```

#### Calculate number of train lines in the station
```Python
# Define a function that counts the number of train lines
def count_train_lines(pt_code):
    if pd.isna(pt_code):
        return 0
    return pt_code.count('/') + 1

# Apply the function to the PT_CODE column and create the TRAIN_LINES column
dfCombinedFirst['TRAIN_LINES'] = dfCombinedFirst['PT_CODE'].apply(count_train_lines)

# Insert TRAIN_LINES next to PT_CODE column
loc = dfCombinedFirst.columns.get_loc('PT_NAME') + 1
dfCombinedFirst.insert(loc, 'TRAIN_LINES', dfCombinedFirst.pop('TRAIN_LINES'))
```
Creating another column to count the number of train lines in the specific station.

#### Key-Value Mapping: Train Codes to unique Key-Value
```Python
train_line_mapping = {
    'EW': 0, # East-West Line
    'CG': 0, # East-West Line to Changi Airport
    'NS': 1, # North-South Line
    'NE': 2, # North-East Line
    'CC': 3, # Circle Line
    'CE': 3, # Circle Line (Bayfront, Marina Bay)
    'DT': 4, # Downtown Line
    'TE': 5, # Thomson-East Coast Line
    'BP': 6, # Bukit Panjang LRT
    'SW': 7, # Sengkang LRT West
    'SE': 7, # Sengkang LRT East
    'PW': 8, # Punggol LRT West
    'PE': 8, # Punggol LRT East
}

def map_train_codes(pt_code):
    # Initialize an empty list to store the mapped train codes
    train_codes = []
    for code in pt_code.split('/'):
        for key in train_line_mapping:
            if code.startswith(key):
                train_codes.append(train_line_mapping[key])
                break 
    return train_codes

# Apply the revised function to the PT_CODE column
dfCombinedFirst['TRAIN_CODES'] = dfCombinedFirst['PT_CODE'].apply(map_train_codes)

# Convert the TRAIN_CODES column to a list
train_codes_list = dfCombinedFirst['TRAIN_CODES'].tolist()

# Insert TRAIN_CODES next to TRAIN_LINES
loc = dfCombinedFirst.columns.get_loc('TRAIN_LINES') + 1
dfCombinedFirst.insert(loc, 'TRAIN_CODES', dfCombinedFirst.pop('TRAIN_CODES'))
```
Here we are mapping each Train Line to a specific unique key-value in a List Format for analysis later.

#### Mapping Latitude & Longitude data to our datas
```Python
# Our location dataset, also available on Kaggle.
dfLocation = pd.read_csv("./datasets/TrainStationLocation.csv")
latitude_mapping = dict(zip(dfLocation['station_name'], dfLocation['lat'])) #Mapping
longitude_mappping = dict(zip(dfLocation['station_name'], dfLocation['lng'])) #Mapping
dfCombinedFirst['PT_LATITUDE'] = dfCombinedFirst['PT_NAME'].map(latitude_mapping) # Creating PT_LATITUDE column
dfCombinedFirst['PT_LONGITUDE'] = dfCombinedFirst['PT_NAME'].map(longitude_mappping) # Creating PT_LONGITUDE column

# Insert PT_LATITUDE & PT_LONGITUDE next to PT_CODE column
loc = dfCombinedFirst.columns.get_loc('PT_CODE') + 1
dfCombinedFirst.insert(loc, 'PT_LATITUDE', dfCombinedFirst.pop('PT_LATITUDE'))
loc1 = dfCombinedFirst.columns.get_loc('PT_CODE') + 2
dfCombinedFirst.insert(loc1, 'PT_LONGITUDE', dfCombinedFirst.pop('PT_LONGITUDE'))
```
Using another dataset that has the latitude & longitude of the station we can map these values to the respective rows.


### As processing Dataset #2 is similar to Dataset #1 there will be no explaination for Dataset #2.

## Results:
#### Before (Dataset #1)
| YEAR_MONTH | DAY_TYPE          | TIME_PER_HOUR | PT_TYPE | PT_CODE | TOTAL_TAP_IN_VOLUME | TOTAL_TAP_OUT_VOLUME |
|------------|-------------------|---------------|---------|---------|---------------------|----------------------|
| 2023-08    | WEEKDAY           | 22            | TRAIN   | NS28    | 752                 | 311                  |
| 2023-08    | WEEKENDS/HOLIDAY  | 22            | TRAIN   | NS28    | 612                 | 223                  |


#### After (Dataset #1)
| DATETIME             | DAY_TYPE | PT_NAME           | TRAIN_LINES | TRAIN_CODES | PT_CODE | PT_LATITUDE | PT_LONGITUDE | TOTAL_TAP_IN_VOLUME | TOTAL_TAP_OUT_VOLUME |
|----------------------|----------|-------------------|-------------|-------------|---------|-------------|--------------|---------------------|----------------------|
| 2023-08-31 22:00:00  | 0        | Marina South Pier | 1           | [1]         | NS28    | 1.271422    | 103.863581   | 752                 | 311                  |
| 2023-08-31 22:00:00  | 1        | Marina South Pier | 1           | [1]         | NS28    | 1.271422    | 103.863581   | 612                 | 223                  |


#### Before (Dataset #2)
| YEAR_MONTH | DAY_TYPE        | TIME_PER_HOUR | PT_TYPE | ORIGIN_PT_CODE | DESTINATION_PT_CODE | TOTAL_TRIPS |
|------------|-----------------|---------------|---------|----------------|---------------------|-------------|
| 2023-08    | WEEKDAY         | 13            | TRAIN   | NE11           | NS19                | 36          |
| 2023-08    | WEEKENDS/HOLIDAY| 13            | TRAIN   | NS19           | NE11                | 11          |


#### After (Dataset #2)
| DATETIME             | DAY_TYPE | ORIGIN_PT_NAME | ORIGIN_TRAIN_LINES | ORIGIN_TRAIN_CODES | ORIGIN_PT_CODE | ORIGIN_PT_LATITUDE | ORIGIN_PT_LONGITUDE | DESTINATION_PT_NAME | DESTINATION_TRAIN_LINES | DESTINATION_TRAIN_CODES | DESTINATION_PT_CODE | DESTINATION_PT_LATITUDE | DESTINATION_PT_LONGITUDE | TOTAL_TRIPS |
|----------------------|----------|----------------|--------------------|--------------------|----------------|--------------------|---------------------|---------------------|-------------------------|-------------------------|---------------------|-------------------------|--------------------------|-------------|
| 2023-08-31 13:00:00  | 0        | Woodleigh      | 1                  | [2]                | NE11           | 1.339202           | 103.870727          | Toa Payoh           | 1                       | [1]                     | NS19                | 1.339202                | 103.870727               | 36          |
| 2023-08-31 13:00:00  | 1        | Toa Payoh      | 1                  | [1]                | NS19           | 1.332405           | 103.847436          | Woodleigh           | 1                       | [2]                     | NE11                | 1.332405                | 103.847436               | 11          |
