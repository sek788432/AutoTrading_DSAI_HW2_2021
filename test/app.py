import pandas as pd
from pandas import Series, DataFrame
import numpy as np
# from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from tensorflow.keras import regularizers
# from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# 解決GPU問題
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def readCSV(x):
    train = pd.read_csv(x, header=None)
    # 去掉nan的值
    # train = train.fillna(144)
    return train


def normalize(train):
    train_norm = train.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    return train_norm


def denormalize(train):
    denorm = train.apply(
        lambda x: x*(np.max(origin_train.iloc[:, 0])-np.min(origin_train.iloc[:, 0]))+np.min(origin_train.iloc[:, 0]))
    return denorm


def buildModel(shape):
    model = Sequential()
    # model.add(GRU(50,input_length=shape[1], input_dim=shape[2],return_sequences=True))
    model.add(GRU(30, input_length=shape[1], input_dim=shape[2], return_sequences=True))
    # model.add(GRU(5,input_length=shape[1], input_dim=shape[2]))
    model.add(Dropout(0.01))

    model.add(GRU(50, return_sequences=True))
    model.add(Dropout(0.01))
    model.add(GRU(70))
    model.add(Dropout(0.01))
    # model.add(GRU(100))
    # model.add(Dropout(0.01))
    # model.add(GRU(5))
    # model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model


def buildTrain(train, past=30, future=1):
    X_train, Y_train, last_dataX = [], [], []
    for i in range(train.shape[0]-future-past):
        X_train.append(np.array(train.iloc[i:i+past]))
        Y_train.append(np.array(train.iloc[i+past:i+past+future][0]))
        if i == train.shape[0]-future-past-1:
            last_dataX.append(np.array(train.iloc[i+1:i+past+1]))
    return np.array(X_train), np.array(Y_train), np.array(last_dataX)


def shuffle(X, Y):
    np.random.seed(10)
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList]


def splitData(X, Y, rate):
    X_train = X[int(X.shape[0]*rate):]
    Y_train = Y[int(Y.shape[0]*rate):]
    X_val = X[:int(X.shape[0]*rate)]
    Y_val = Y[:int(Y.shape[0]*rate)]
    return X_train, Y_train, X_val, Y_val


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--training',
                        default='training.csv',
                        help='input training data file name')
    parser.add_argument('--testing',
                        default='testing.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()
    # ============================訓練===========================
    train = readCSV(args.training)
    # train = readCSV("training.csv")
    #train = pd.read_csv("training.csv",header=None)
    train_norm = normalize(train)
    X_train, Y_train, Last_dataX = buildTrain(train_norm, 30, 1)
    X_train, Y_train = shuffle(X_train, Y_train)
    X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, 0.1)
    model = buildModel(X_train.shape)
    callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
    model.fit(X_train, Y_train, epochs=100, batch_size=32,
              validation_data=(X_val, Y_val), callbacks=[callback])
    # model.save('test.h5')
    # ==============================================================
    # model = load_model('test.h5')
    testing = readCSV(args.testing)
    origin_train = train
    hat_action = []  # 動作
    have = 0  # 起始沒有股票
    y_pre = []  # 預測y
    gap = 0.5  # 兩天股票差
    print("waiting for action")
    # =============================ACTION==============================
    for i in range(len(testing.iloc[:, 0])):
        # print(hat_action)
        if have == -1:  # 若前一天要求今天賣空 則紀錄今天的股價
            money = testing.iloc[i][0]
        # pred = pd.DataFrame(pred)
        train = train.append(testing.iloc[i])  # 將今天資料加入training data
        train_norn = normalize(train)
        X_train, Y_train, Last_dataX = buildTrain(train_norn, 30, 1)  # 提出最後三十筆，包含新的第一天
        pred = model.predict(Last_dataX)  # 得到明天開盤預測
        pred = pd.DataFrame(pred)
        a = denormalize(pred)
        y_pre.append(a.iloc[0][0])
        # print(np.sqrt((a-y_test[i])**2))
        if i < len(testing.iloc[:, 0])-1:
            if a[0][0] > testing.iloc[i][0]:  # 明天>今天
                if have == 1:
                    if a[0][0]-testing.iloc[i][0] >= gap:
                        hat_action.append(-1)
                        have = 0
                        continue
                    if a[0][0]-testing.iloc[i][0] < gap:
                        hat_action.append(0)
                        have = 1
                        continue
                if have == 0:
                    if a[0][0]-testing.iloc[i][0] >= gap:
                        hat_action.append(-1)
                        have = -1
                        continue
                    if a[0][0]-testing.iloc[i][0] < gap:
                        hat_action.append(0)
                        have = 0
                if have == -1:
                    hat_action.append(0)
                    have = -1
                    continue
            if a[0][0] < testing.iloc[i][0]:  # 今天>明天
                if have == 1:
                    if testing.iloc[i][0]-a[0][0] >= gap:
                        hat_action.append(-1)
                        have = 0
                        continue
                    if testing.iloc[i][0]-a[0][0] < gap:
                        hat_action.append(0)
                        have = 1
                        continue
                if have == 0:
                    if testing.iloc[i][0]-a[0][0] >= gap:
                        hat_action.append(1)
                        have = 1
                        continue
                    if testing.iloc[i][0]-a[0][0] < gap:
                        hat_action.append(0)
                        have = 0
                        continue
                if have == -1:
                    if money < a[0][0]:  # 若賣空價格小於明天預測開盤，不動作
                        hat_action.append(0)
                        continue
                    if money > a[0][0]:  # 若賣空價格大於明天預測開盤，則買進賺價差
                        hat_action.append(1)
                        have = 0
                        continue
            if a[0][0] == testing.iloc[i][0]:
                hat_action.append(0)
                continue
    # ============================================================================
    with open(args.output, 'w') as output_file:
        for row in range(len(hat_action)):
            # We will perform your action as the open price in the next day.
            action = hat_action[row]
            output_file.write(str(action)+"\n")
    print("done")
