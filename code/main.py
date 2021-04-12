from trader import Trader
from model import Model
import sys
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                        default='training_data.csv',
                        help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()

    # Model Type
    model_type = "lstm"
    model = Model(args.training, args.testing)
    model.execute(model_type)
    if model_type == "xgboost":
        Trader(args.training, args.testing, args.output).run_xgb(model)
    elif model_type == "lstm":
        Trader(args.training, args.testing, args.output).run_lstm(model)
    else:
        sys.exit("Model Not Found")

