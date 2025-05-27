import os
import pickle
import click
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

mlflow.sklearn.autolog()


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)



def run_train():

    data_path = "./output"
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-experiment") # sets the experiment
    mlflow.sklearn.autolog(log_model_signatures=False, log_input_examples=False) 

    with mlflow.start_run(): # Wrap training code

     X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
     X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    rf = RandomForestRegressor(max_depth=10, random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)

    rmse = root_mean_squared_error(y_val, y_pred, squared=False)


if __name__ == '__main__':
    run_train()
