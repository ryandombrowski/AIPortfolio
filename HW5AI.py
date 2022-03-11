import pandas as pd
from pandas import DataFrame
import numpy as np
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
import csv


def one_hot(df, cols):
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
    return df


# Method for encoding is OneHot encoding, as no ordinal relationship exists between the features listed
file = 'cars.csv'  # establish file to be read
carinfo = pd.read_csv(file)         # reads file's data across all rows and columns
dataset = carinfo.values
pd.set_option('display.max_columns', None)

carinfo.head()
print(carinfo.dtypes)       # all data is in an object format. Good on that
print(dataset)

carinfo[carinfo.isnull().any(axis=1)]   # find null values in the data set
print(carinfo[carinfo.isnull().any(axis=1)])    # shows where the null values were

# changes NaN to electric motor for cars that run on electricity
# choose to replace data so none is wasted
carinfo['engine_type'] = carinfo['engine_type'].replace(np.nan, 'electric motor')
carinfo['engine_capacity'] = carinfo['engine_capacity'].replace(np.nan, 0)  # replaced null values with 0 for electric cars
print(carinfo)

print(carinfo[carinfo.isnull().any(axis=1)])
# List of null values is now empty

print(carinfo.describe())       # Gives initial description of the dataset

# Once the data is all clear of NaN values, the translation to binned ranges begins

binwidth = int((max(carinfo['odometer_value'])-min(carinfo['odometer_value']))/3)
bins = range(int(min(carinfo['odometer_value'])), int(max(carinfo['odometer_value'])), binwidth)
carinfo['odometer_value'] = pd.cut(carinfo['odometer_value'], bins, labels=["Low", "Mid", "High"])
carinfo['odometer_value'] = carinfo['odometer_value'].cat.codes

# Establish the columns within the dataframe that need to be changed from categorical data
features_to_encode = ['manufacturer_name', 'model_name', 'transmission', 'color',
                      'year_produced', 'engine_fuel', 'engine_has_gas', 'engine_type', 'body_type', 'has_warranty',
                      'state', 'drivetrain', 'is_exchangeable', 'location_region', 'feature_0',
                      'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6',
                      'feature_7', 'feature_8', 'feature_9']

print(one_hot(carinfo, features_to_encode))
