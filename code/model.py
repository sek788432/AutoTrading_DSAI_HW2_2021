import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import sys
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt

class Model():
    def __init__(self, train_data_path, test_data_path):
        # model varaibles
        self.train_x = None
        self.train_y = None
        self.combine_x = None
        self.model = None

        # normalized scale
        self.scale_ = None

        # full  data
        self.train = None
        self.test = None
        self.combine = None

        #data path
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path

        #LSTM variables
        self.n_past = 300    # Number of past days we want to use to predict the future
        self.n_future = 1    # Number of days we want to predict into the future
        self.units = 1024    # LSTM Base Units
        self.epochs = 65     # Epochs
        self.lr  = 1e-3      # Base Learning Rate

    def load_data(self):
        self.train = pd.read_csv(self.train_data_path, header=None)
        self.train.columns = ['open', 'high', 'low', 'close']
        self.test = pd.read_csv(self.test_data_path, header=None)
        self.test.columns = ['open', 'high', 'low', 'close']
        self.combine = pd.concat([self.train, self.test])
        self.combine = self.combine.reset_index(drop=True)

    def execute(self, model_type = "lstm"):
        self.load_data()
        if model_type == "xgboost":
            self.xgb()
        elif model_type == "lstm":
            self.lstm()
        else:
            sys.exit("Model Not found")

    def xgb(self):
        self.preprocess_xgb(n_day_mean=5, sliding_window=1)
        self.model = XGBRegressor(objective="reg:squarederror", n_estimators=1000)
        self.model.fit(self.train_x, self.train_y)

    def lstm(self):
        # set GPU
        gpu_device = tf.config.experimental.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(gpu_device[0], True)
        # preprocess
        self.preprocess_lstm()
        
        # build model
        self.model = Sequential()
        self.model.add(LSTM(units=self.units, return_sequences = True, 
                       kernel_initializer = 'glorot_uniform', 
                       input_shape  =  (self.train_x.shape[1], self.train_x.shape[2])))
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(units = self.units, kernel_initializer = 'glorot_uniform', return_sequences = True))
        self.model.add(Dropout(0.2))
        self.units = int(self.units/2)
        self.model.add(LSTM(units = self.units, kernel_initializer = 'glorot_uniform', return_sequences = True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units = self.units, kernel_initializer = 'glorot_uniform', return_sequences = True))
        self.model.add(Dropout(0.2))
        self.units = int(self.units/2)
        self.model.add(LSTM(units = self.units, kernel_initializer = 'glorot_uniform', return_sequences = True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units = self.units, kernel_initializer = 'glorot_uniform'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1, activation='linear'))
        self.model.summary()
        self.train_model()

    def train_model(self):
        callback = EarlyStopping(monitor='val_mae', patience=12, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=6, verbose=1, min_lr=1e-6)
        opt = Adam(learning_rate= self.lr)
        self.model.compile(optimizer=opt, loss='mse', metrics=['mae'])
        history = self.model.fit(self.train_x, self.train_y,
                            callbacks=[callback, reduce_lr],
                            epochs=self.epochs,
                            validation_split=0.1,
                            shuffle=True)

    def predict_lstm(self):
        test_len = len(self.test)
        features = self.combine_x.shape[2]
        forecast = self.model.predict(self.combine_x[-test_len - 1:-1])  # forecast
        forecast_copies = np.repeat(forecast, features, axis=-1)
        pred = self.scale_.inverse_transform(forecast_copies)[:, 0]
        actual = self.test['open']
        
        #plot actual data and predict data
        plt.plot(actual, label = "actual")
        plt.plot(pred, label = "predict")
        plt.xlabel('x - axis')
        plt.ylabel('y - axis')
        plt.title('actual vs predict')
        my_x_ticks = np.arange(0, 20, 1)
        plt.xticks(my_x_ticks)
        plt.legend()
        plt.grid()
        plt.savefig("../output_file/pred_vs_actual.png")
        return pred

    def preprocess_lstm(self):
        df_for_training = self.train[['open']].copy()
        df_for_testing = self.test[['open']].copy()
        df_for_combine = self.combine[['open']].copy()

        # Normalization
        self.scale_ = StandardScaler()
        self.scale_ = self.scale_.fit(df_for_combine)
        df_for_training_scaled = self.scale_.transform(df_for_training)
        df_for_combine_scaled = self.scale_.transform(df_for_combine)
        df_for_testing_scaled = self.scale_.transform(df_for_testing)
        self.make_lstm_input(df_for_training_scaled, df_for_combine_scaled)

    def make_lstm_input(self, df_for_training_scaled,
                        df_for_combine_scaled):
        trainX = []
        trainY = []
        np_combine = []

        for i in range(self.n_past, len(df_for_training_scaled) - self.n_future + 1):
            trainX.append(df_for_training_scaled[i - self.n_past:i, 0:df_for_training_scaled.shape[1]])
            trainY.append(df_for_training_scaled[i + self.n_future - 1:i + self.n_future, 0])  # first column
        for i in range(self.n_past, len(df_for_combine_scaled) - self.n_future + 1):
            np_combine.append(df_for_combine_scaled[i - self.n_past:i, 0:df_for_combine_scaled.shape[1]])

        self.train_x, self.train_y = np.array(trainX), np.array(trainY)
        self.combine_x = np.array(np_combine)
        print('trainX shape == {}.'.format(self.train_x.shape))
        print('trainY shape == {}.'.format(self.train_y.shape))
        print('combineX shape == {}.'.format(self.combine_x.shape))

    def predict_xgb(self, today_price):
        return self.model.predict(np.array([[today_price]]))[0]

    def preprocess_xgb(self, n_day_mean, sliding_window):
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