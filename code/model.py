import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


class Model():
    def __init__(self, train_data_path):
        self.train_x = None
        self.train_y = None
        self.model = None
        self.train = None
        self.train_data_path = train_data_path

    def load_data(self):
        self.train = pd.read_csv(self.train_data_path, header=None)
        self.train.columns = ['open', 'high', 'low', 'close']

    def pre_process(self):
        # copy train data
        train_set = self.train[['open']].copy()
        day_shift = 5
        col_name = []
        # generate training data
        # make last 5 days as X and the next day as Y
        for i in range(1, day_shift + 1):
            train_set[str(i)] = train_set.open.shift(-i)
        train_set.dropna(inplace=True)
        # change dataframe cloumns
        for i in range(1, day_shift + 1):
            col_name.append("Day_" + str(i))
        col_name.append('y')
        train_set.columns = col_name

        train_val = train_set.values
        self.train_x = train_val[:, :-1]
        self.train_y = train_val[:, -1]

    def train_model(self):
        self.load_data()
        self.pre_process()
        self.model = XGBRegressor(objective="reg:squarederror", n_estimators=1000)
        self.model.fit(self.train_x, self.train_y)

    def predict(self, five_days):
        return self.model.predict(np.array([five_days]))[0]
