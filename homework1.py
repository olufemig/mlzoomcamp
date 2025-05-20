import numpy as np
import pandas as pd
import datetime as dt
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error

#downloading the data

df = pd.read_parquet('./data/yellow_tripdata_2023-01.parquet', engine='pyarrow')
val_df = pd.read_parquet('./data/yellow_tripdata_2023-02.parquet', engine='pyarrow')
#df.info()
num_columns = len(df.columns)
print("Number of columns:", num_columns)

#computing duration and its standard deviation
df["tpep_pickup_datetime"] = pd.to_datetime(df.tpep_pickup_datetime)
df["tpep_dropoff_datetime"] = pd.to_datetime(df.tpep_dropoff_datetime)
df["duration"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds()/60
total_count = len(df)
std_dev = df['duration'].std()
print("Standard Deviation:", std_dev)

#dropping outliers and determining fraction of whole dataset
filtered_df = df[(df['duration'] >= 1) & (df['duration'] <= 60)]
filtered_count = len(filtered_df)
percent_left = (filtered_count/total_count) * 100
print("percentage_after_filtering:", percent_left)

#one hot encoding
categorical = ['PULocationID', 'DOLocationID']
filtered_df = filtered_df.copy()
filtered_df[categorical] = filtered_df[categorical].astype(str)
train_dicts = filtered_df[categorical].to_dict(orient='records')
dv = DictVectorizer()

#training model and calculating rmse
X_train = dv.fit_transform(train_dicts)
print(X_train.shape)

target = 'duration'
y_train = filtered_df[target].values
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_train)

rmse = root_mean_squared_error(y_train, y_pred)
print("rmse:", rmse)

#rmse for validation dataset
val_df["tpep_pickup_datetime"] = pd.to_datetime(val_df.tpep_pickup_datetime)
val_df["tpep_dropoff_datetime"] = pd.to_datetime(val_df.tpep_dropoff_datetime)
val_df["duration"] = (val_df["tpep_dropoff_datetime"] - val_df["tpep_pickup_datetime"]).dt.total_seconds()/60

val_df = val_df[(val_df['duration']>=1) & (val_df['duration']<=60)]
val_df[categorical] = val_df[categorical].astype(str)
valid_dicts = val_df[categorical].to_dict(orient='records')
X_valid = dv.transform(valid_dicts)

y_valid = val_df[target].values
y_pred_valid = lr.predict(X_valid)

val_rmse = root_mean_squared_error(y_valid, y_pred_valid)
print("val rmse:", rmse)