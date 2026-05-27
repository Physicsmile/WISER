from torch.cuda import set_device
import sys
import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from parse_arguments import parse_arguments
from experiments import Experiment
import pandas as pd
import wandb
os.environ["TOKENIZERS_PARALLELISM"] = "false"

config_file = '../config/start_config_circo.json'
def main(args):
    args = parse_arguments()
    Experiment(args=args).run()
def json_config_start(args):
    config_data = pd.read_json(config_file)
    config_data = config_data.iloc[0]
    config_data = config_data.to_dict()

    for key, value in config_data.items():
        setattr(args, key, value)
    Experiment(args=args).run()

if __name__ == '__main__':
    args = parse_arguments()
    json_config_start(args)