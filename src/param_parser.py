"""Parsing the parameters."""

import argparse

def parameter_parser():

    parser = argparse.ArgumentParser(description="Run GWNN.")

    parser.add_argument("--data_path",
                    nargs="?",
                    default="C:/Users/puser/OneDrive - postech.ac.kr/project/Graph Generation/code/GraphGenerator/input",
                help="top data path.")

    parser.add_argument("--log_path",
                    nargs="?",
                    default="C:/Users/puser/OneDrive - postech.ac.kr/project/Graph Generation/code/GWVAE",
                help="top data path.")

    parser.add_argument("--log-path",
                        nargs="?",
                        default="./logs/enzyme_logs.json",
	                help="Log json.")

    parser.add_argument("--epochs",
                        type=int,
                        default=70,
	                help="Number of training epochs. Default is 200.")

    parser.add_argument("--filters",
                        type=int,
                        default=32,
	                help="Filters (neurons) in convolution. Default is 32.")

    parser.add_argument("--batch_size",
                        type=int,
                        default=16,
	                help="Number of data for one batch. Default is 8.")

    parser.add_argument("--test_size",
                        type=float,
                        default=0.2,
	                help="Ratio of training samples. Default is 0.2.")

    parser.add_argument("--dropout",
                        type=float,
                        default=0.05,
	                help="Dropout probability. Default is 0.05.")

    parser.add_argument("--seed",
                        type=int,
                        default=42,
	                help="Random seed for sklearn pre-training. Default is 42.")

    parser.add_argument("--nscale",
                        type=int,
                        default=5,
	                help="number of scales. Default is 1.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.03,
	                help="Learning rate. Default is 0.01.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=10**-5,
	                help="Adam weight decay. Default is 10^-5.")

    parser.add_argument("--num_train_data",
                        type=int,
                        default=480,
	                help="Learning rate. Default is 0.01.")

    return parser.parse_args()
