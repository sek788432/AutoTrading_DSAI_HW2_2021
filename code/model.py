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

    def preprocess_n_mean(self, n_day_mean, sliding_window):
        # copy train data
        train_set = self.train[['open']].copy()

        # build n day mean data
        x = []
        open_price = train_set.iloc[:, 0]
        i = n_day_mean
        while i <= len(open_price):
            sum_ = 0
            for j in range(i - n_day_mean, i):
                sum_ += open_price[j]
            x.append(sum_ / n_day_mean)
            i += sliding_window
        y = []
        for i in range(1, len(x)):
            y.append(x[i])
        x.pop()
        train_set = pd.DataFrame(x, columns=['last_day'])
        train_set['y'] = y

        train_val = train_set.values
        self.train_x = train_val[:, :-1]
        self.train_y = train_val[:, -1]

    def train_model(self):
        self.load_data()
        self.preprocess_n_mean(n_day_mean=5, sliding_window=1)
        self.xgb()

    def predict(self, five_days):
        return self.model.predict(np.array([five_days]))[0]

    def predict_mean(self, today_price):
        return self.model.predict(np.array([[today_price]]))[0]

    def xgb(self):
        self.model = XGBRegressor(objective="reg:squarederror", n_estimators=1000)
        self.model.fit(self.train_x, self.train_y)

    def lstm(self):
        pass
