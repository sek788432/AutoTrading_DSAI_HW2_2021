import pandas as pd
import csv
import sys
from model import Model


class Trader:
    def __init__(self, train_path, test_path, output_path):
        self.buy = 1
        self.hold = 0
        self.sell = -1
        # stock contains number of stocks hold, -1~1
        self.stock = 0
        self.train_path = train_path
        self.test_path = test_path
        self.output_path = output_path
        # last five data from training data
        self.last_five_data = [151.95, 152.06, 152.35, 152.81, 153.65]
        self.output = []

    def Trade(self, today_price, tomorrow_price):
        action = None
        if tomorrow_price > today_price:
            if self.stock == 0:
                action = self.buy
                self.stock += 1
            elif self.stock == 1:
                action = self.hold
            elif self.stock == -1:
                action = self.buy
                self.stock += 1
            else:
                sys.exit("There's some mistake!!")
        else:
            if self.stock == 0:
                action = self.sell
                self.stock -= 1
            elif self.stock == 1:
                action = self.sell
                self.stock -= 1
            elif self.stock == -1:
                action = self.hold
            else:
                sys.exit("There's some mistake!!")
        self.output.append(action)

    def make_output(self):
        with open(self.output_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerows(map(lambda x: [x], self.output))
        f.close()

    def run_normal(self, model):
        # read test data
        df = pd.read_csv(self.test_path, header=None)
        tomorrow_price = None
        # handle everyday open price
        for ind, open_price in enumerate(df.loc[:, 0]):
            if ind == 19:
                break
            print("Actual:", open_price, " Predict:", tomorrow_price)
            # update last five data from today
            self.last_five_data.append(open_price)
            self.last_five_data.pop(0)
            # call model to predict tomorrow price
            tomorrow_price = model.predict(self.last_five_data)
            self.Trade(open_price, tomorrow_price)

        # make submission data
        self.make_output()

    def run_mean(self, model):
        # read test data
        df = pd.read_csv(self.test_path, header=None)
        tomorrow_price = None
        open_price = df.loc[:, 0]
        df = self.preprocess_mean(open_price, n_day_mean=5, sliding_window=1)
        print(df.shape)
        print(type(df))
        # handle everyday open price
        for ind, open_price in enumerate(df):
            print("Actual:", open_price, " Predict:", tomorrow_price)
            # update last five data from today

            # call model to predict tomorrow price
            tomorrow_price = model.predict_mean(open_price)
            self.Trade(open_price, tomorrow_price)

        # make submission data
        self.make_output()

    def preprocess_mean(self, open_price, n_day_mean, sliding_window):
        train = pd.read_csv(self.train_path, header=None)
        train.columns = ['open', 'high', 'low', 'close']
        last = train[['open']][-(n_day_mean-1):].values
        last = last.squeeze().tolist()
        last.reverse()
        for i in last:
            open_price.loc[-1] = i  # adding a row
            open_price.index = open_price.index + 1  # shifting index
            open_price = open_price.sort_index()  # sorting by index

        # build n day mean data

        x = []
        # open_price = test_set.iloc[:, 0]
        i = n_day_mean
        while i <= len(open_price):
            sum = 0
            for j in range(i - n_day_mean, i):
                sum += open_price[j]
            x.append(sum / n_day_mean)
            i += sliding_window

        y = []
        for i in range(1, len(x)):
            y.append(x[i])
        x.pop()
        test_set = pd.DataFrame(x, columns=['last_day'])
        test_set['y'] = y
        return test_set['last_day']
        # test_x = test_val[:,:-1]
        # test_y = test_val[:, -1]

    def run_test(self):
        # read test data
        df = pd.read_csv(self.test_path, header=None)
        tomorrow_price = None
        # handle everyday open price
        for ind, open_price in enumerate(df.loc[:, 0]):
            if ind == 19:
                break
            print("Actual:", open_price, " Predict:", tomorrow_price)
            # update last five data from today
            self.last_five_data.append(open_price)
            self.last_five_data.pop(0)
            # call model to predict tomorrow price
            tomorrow_price = sum(self.last_five_data) / 5
            self.Trade(open_price, tomorrow_price)

        # make submission data
        self.make_output()


if __name__ == "__main__":
    Trader("../data/testing.csv").run()
