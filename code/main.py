from trader import Trader
from model import Model
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
    md = Model(args.training)
    md.train_model()
    Trader(args.training, args.testing, args.output).run_mean(md)
    # Trader(args.training, args.testing, args.output).run_test()
