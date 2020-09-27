import argparse
import json
import logging
import torch
import numpy as np
import random

from algorithms.Baseline.src.Trainer_Baseline import Trainer_Baseline
from algorithms.AGG.src.Trainer_AGG import Trainer_AGG
from algorithms.JiGen.src.Trainer_JiGen import Trainer_JiGen
from algorithms.Metric.src.Trainer_Metric import Trainer_Metric
from algorithms.DSDI.src.Trainer_DSDI import Trainer_DSDI

def set_random_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.cuda.manual_seed(seed_value)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

algorithms_map = {
    'Baseline': Trainer_Baseline,
    'JiGen': Trainer_JiGen,
    'AGG': Trainer_AGG,
    'Metric': Trainer_Metric,
    'DSDI': Trainer_DSDI
}

if __name__ == "__main__":
    # torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", help = "Path to configuration file")
    args = parser.parse_args()
    with open(args.config, "r") as inp:
        args = argparse.Namespace(**json.load(inp))
    
    set_random_seed(args.seed_value)
    logging.basicConfig(filename = "algorithms/" + args.algorithm + "/results/logs/" + args.exp_name + '.log', filemode = 'w', level = logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    trainer = algorithms_map[args.algorithm](args, device)
    trainer.train()
    trainer.test()
    print("Finished!")