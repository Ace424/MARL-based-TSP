import os
import time
import torch
import logging
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import argparse
from config import configs
from utils import log
from trainer import trainPPO, sample, trainDQN
from scipy.io import loadmat
from sklearn.neighbors import LocalOutlierFactor as LOF
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from gcn import Feature_GCN
from dqn import DQN

MEMORY_CAPACITY = 20  # The size of experience replay buffer
EXPLORE_STEPS = 30  # How many exploration steps you'd like, should be larger than MEMORY_CAPACITY


# def main(args, df, c_columns, d_columns, target, mode, model, metric):
def main(args, dataset_path, target, covariates, mode, model, metric):
    # 新建日志
    times = time.strftime('%Y%m%d-%H%M')
    log_dir = fr"./logs/train/{args.file_name}_{times}"
    log(log_dir)
    logging.info(f"args={args}")
    trainDQN(args, dataset_path, target, covariates, mode, model, metric)
    # trainPPO(args, df, c_columns, d_columns, target, mode, model, metric)


if __name__ == '__main__':
    ############################
    file_name = 'car_sale_prediction'

    parser_ppo = argparse.ArgumentParser()
    # data
    parser_ppo.add_argument("--target", type=str, default=None)
    parser_ppo.add_argument("--train_size", type=float, default=0.8)
    # ppo
    parser_ppo.add_argument("--epochs", type=int, default=50)
    parser_ppo.add_argument("--ppo_epochs", type=int, default=5)
    parser_ppo.add_argument("--episodes", type=int, default=24)
    parser_ppo.add_argument("--lr", type=float, default=1e-3)
    parser_ppo.add_argument("--entropy_weight", type=float, default=1e-4)
    parser_ppo.add_argument("--baseline_weight", type=float, default=0.95)
    parser_ppo.add_argument("--gama", type=float, default=0.9)
    parser_ppo.add_argument("--gae_lambda", type=float, default=0.95)
    parser_ppo.add_argument("--batch_size", type=int, default=4)
    parser_ppo.add_argument("--d_model", type=int, default=64)
    parser_ppo.add_argument("--d_k", type=int, default=32)
    parser_ppo.add_argument("--d_v", type=int, default=32)
    parser_ppo.add_argument("--d_ff", type=int, default=128)
    parser_ppo.add_argument("--n_heads", type=int, default=6)
    parser_ppo.add_argument("--worker", type=int, default=24)
    parser_ppo.add_argument("--steps_num", type=int, default=2)

    parser_ppo.add_argument("--device", type=str, default="cpu")
    parser_ppo.add_argument("--seed", type=int, default=2)
    parser_ppo.add_argument("--cv", type=int, default=1)
    parser_ppo.add_argument("--cv_train_size", type=float, default=0.7)
    parser_ppo.add_argument("--cv_seed", type=int, default=42)
    parser_ppo.add_argument("--split_train_test", type=str, default=False)
    parser_ppo.add_argument("--mode", type=str, default=None, help="classify or regression")
    parser_ppo.add_argument("--model", type=str, default="rf", help="lr or xgb or rf")
    parser_ppo.add_argument("--metric", type=str, default="auc", help="f1,ks,auc,r2")
    parser_ppo.add_argument("--file_name", type=str, default=file_name)

    args = parser_ppo.parse_args()

    data_configs = configs[args.file_name]
    # c_columns = data_configs['c_columns']
    # d_columns = data_configs['d_columns']
    target = data_configs['target']
    covariates = data_configs['covariates']
    dataset_path = data_configs["dataset_path"]

    # 如果没有在命令行执行指定参数，则会根据指定的filename从config_pool.py里获取相应数据集的信息
    if args.mode:
        mode = args.mode
    else:
        mode = data_configs['mode']
    if args.model:
        model = args.model
    else:
        model = data_configs["model"]
    if args.metric:
        metric = args.metric
    else:
        metric = data_configs["metric"]

    # 固定随机种子
    seed = 0
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # df = pd.read_csv(dataset_path + target)
    # main(args, df, c_columns, d_columns, target, mode, model, metric)
    main(args, dataset_path, target, covariates, mode, model, metric)
