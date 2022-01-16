import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scipy.io import loadmat
from config import configs
import scipy.stats as st
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor as LOF
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, ShuffleSplit, cross_val_score, KFold, \
    StratifiedKFold
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, r2_score, roc_auc_score, roc_curve, auc, mean_squared_error, mean_absolute_error

import pickle
import os
import logging
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import Counter
from model_eval import *
from feature_engineering import utils_memory
from feature_engineering import feature_generate_memory
from feature_engineering.pipline_memory_with_selector import Pipline, generate_fes
from feature_engineering.memory_no_encode import Memory
from operations import OPS


# TODO: 封装一些功能函数
def rae(y_true: np.ndarray, y_pred: np.ndarray):
    up = np.abs(y_pred - y_true).sum()
    down = np.abs(y_true.mean() - y_true).sum()
    score = 1 - up / down
    return score


def get_ops(n_features):
    ops = []
    arithmetic = OPS["arithmetic"]
    value_convert = OPS["value_convert"]
    ### Q1：关于二元算子op的生成，贪心？？
    for i in range(1, 6):
        op = arithmetic[i - 1]
        for j in range((i - 1) * n_features, i * n_features):
            ops.append(op)
    ops.extend(value_convert)
    ops.append("None")
    return ops


def remove_duplication(data):
    _, idx = np.unique(data, axis=1, return_index=True)
    y = data[:, np.sort(idx)]
    return y, np.sort(idx)


def calculate_mutual(x, y, mode):
    if mode == "classify":
        mic = mutual_info_classif(x, y, random_state=42)
        return mic
    elif mode == "regression":
        mir = mutual_info_regression(x, y, random_state=42)
        return mir


def calculate_statistics(data: np.ndarray, label: np.ndarray, mode):
    min = np.min(data, axis=0)
    max = np.max(data, axis=0)
    quantile = np.percentile(data, (1, 25, 50, 75, 99), axis=0)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    skewness = st.skew(data, axis=0)
    kurtosis = st.kurtosis(data, axis=0)
    encode = np.vstack((quantile, mean, std, skewness, kurtosis))
    encode = (encode - min) / (max - min)
    # mutal = calculate_mutual(data,label,mode)
    # encode = np.vstack((encode,mutal))
    return encode


def label_encode_to_onehot(data, data_train=None):
    """
    :param x:np.array
    :return:one_hot
    """
    onehot = None
    for col in range(data.shape[1]):
        if isinstance(data_train, np.ndarray):
            x = data[:, col]
            x_ = data_train[:, col]
            num_class = int(np.max(x_) + 1)
        else:
            x = data[:, col]
            num_class = int(np.max(x) + 1)
        m = np.zeros((len(x), num_class))
        m[range(len(x)), x] = 1
        if not isinstance(onehot, np.ndarray):
            onehot = m
        else:
            onehot = np.concatenate((onehot, m), axis=1)

    return onehot


def split_train_test(df, args, d_columns, target, mode):
    if mode == "classify":
        if args.split_train_test:
            # 留待悬念，test去哪儿了？
            df_train_val, test = train_test_split(df, train_size=args.train_size, random_state=args.seed,
                                                  stratify=df[target])
        else:
            df_train_val = df

        # 合并小于数据集数量千分之一的类别
        df_train_val = df_train_val.copy()
        for col in d_columns:
            new_fe = merge_categories(df_train_val[col].values)
            df_train_val[col] = new_fe

    else:
        if args.split_train_test:
            df_train_val, test = train_test_split(df, train_size=0.8, random_state=args.seed, )
        else:
            df_train_val = df
        df_train_val = df_train_val.copy()
        for col in d_columns:
            new_fe = merge_categories(df_train_val[col].values)
            df_train_val[col] = new_fe

    # df_train_val.drop_duplicates(keep="first", inplace=True)
    # df_train_val.reset_index(drop=True,inplace=True)
    return df_train_val


def cluster_data(name, df_train_val, ratio=2):
    file_name = name
    c_columns = configs[file_name]["c_columns"]
    d_columns = configs[file_name]["d_columns"]
    target = configs[file_name]["target"]
    df_process = pd.DataFrame()
    for column in df_train_val.columns.values:
        if column in c_columns:
            df_process[column] = (df_train_val[column].values - df_train_val[column].values.mean()) / (
                df_train_val[column].values.std())
        elif column in d_columns:
            df_process[column] = utils_memory.categories_to_int(df_train_val[column].values)
        else:
            df_process[target] = utils_memory.categories_to_int(df_train_val[column].values)

    df_train_1 = df_process[df_process.iloc[:, -1] == 1]
    df_train_0 = df_process[df_process.iloc[:, -1] == 0]
    nums_1 = len(df_train_1)
    df_train_0_ = df_train_0.copy()
    df_cluster = pd.DataFrame()
    gmm = GaussianMixture(n_components=5, random_state=42)
    gmm.fit(df_train_0_.iloc[:, 0:-1])
    labels = gmm.predict(df_train_0_.iloc[:, 0:-1])

    df_train_0_["labels"] = labels
    df_discard = pd.DataFrame()
    # print(df_train_0_["labels"].value_counts())
    for j in range(5):
        data_init = df_train_0_[df_train_0_.iloc[:, -1] == j].sample(frac=1)
        data = data_init.iloc[0:ratio * nums_1, 0:-1]
        data_discard = data_init.iloc[ratio * nums_1:, 0:-1]
        df_discard = pd.concat((df_discard, data_discard), axis=0)
        df_cluster = pd.concat((df_cluster, data), axis=0)
    df_train = pd.concat((df_cluster, df_train_1), axis=0).sample(frac=1).reset_index(drop=True)

    df_train_index = []
    df_discard_index = []
    for i, value in enumerate(df_process.values):
        ret = (value == df_train.values).all(1).any()
        if ret:
            df_train_index.append(i)
        else:
            df_discard_index.append(i)
    return np.array(df_train_index), np.array(df_discard_index)


def get_xy(pipline_args_train, c_ops, d_ops):
    memory = Memory()
    pipline_args_train["memory"] = memory
    c_fes_train, d_fes_train, label_train = generate_fes(c_ops, d_ops, pipline_args_train)

    if c_ops == [{}]:
        x_train = d_fes_train
        y_train = label_train

    elif d_ops == [{}]:
        x_train = c_fes_train
        y_train = label_train

    else:
        x_train = np.concatenate((c_fes_train, d_fes_train), axis=1)
        y_train = label_train
    return x_train, y_train


def load_model_machine(model_dir):
    """加载"""
    models = []
    for file in os.listdir(model_dir):
        file_path = os.path.join(model_dir, file)
        with open(file_path, "rb") as fr:
            model = pickle.load(fr)
            models.append(model)
    return models


def log(exp_dir):
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
    fh = logging.FileHandler(os.path.join(exp_dir, "log.txt"))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)


def merge_categories(col, threshold=0.001):
    nums = max(int(len(col) * threshold), 5)
    # nums = 5
    count = dict(Counter(col))
    sorted_count = dict(sorted(count.items(), key=lambda x: x[1], reverse=True))
    replaec_dict = {}
    replace = False
    merge_name = None
    for name in sorted_count.keys():
        if replace:
            replaec_dict[name] = merge_name
        else:
            if sorted_count[name] < nums:
                merge_name = name
                replace = True
    new_fe = [replaec_dict[x] if x in replaec_dict else x for x in col]
    return new_fe


def features_process(df_train_val, mode, c_columns, d_columns, target):
    # 分类任务的label数值化
    if mode == "classify":
        col = df_train_val[target].values.reshape(-1)
        df_train_val[target] = utils_memory.categories_to_int(col)
    df_t = df_train_val[target].reset_index(drop=True)  # 数值化后的df_target列

    # 连续数据编码
    df_c_encode = pd.DataFrame()
    if len(c_columns):
        for col in c_columns:
            df_c_encode[col] = feature_generate_memory.normalization(df_train_val[col].values).reshape(-1)

        df_c_encode = pd.concat((df_c_encode, df_t), axis=1)

    else:
        df_c_encode = pd.DataFrame()

    # 离散数据编码
    df_d_labelencode = pd.DataFrame()
    for column in d_columns:
        df_d_labelencode[column] = utils_memory.categories_to_int(df_train_val[column].values)

    return df_d_labelencode, df_c_encode, df_t


def feature_merge(c_op, d_op, pipline_args_train, scores_b, metric):
    # 通过pipeline生成特征
    x, y = get_xy(pipline_args_train, c_op, d_op)
    x_train = x
    y_train = y

    if pipline_args_train["mode"] == "regression":
        model_xgb = xgb_regression()
    else:
        model_xgb = xgb_classify()

    my_cv = StratifiedShuffleSplit(n_splits=2, train_size=0.5, random_state=8)

    if len(np.unique(y)) > 2:
        eval_metric = "mlogloss"
    else:
        eval_metric = "logloss"
    scores = []
    for index_train, index_test in my_cv.split(x_train, y_train):
        train_x = x_train[index_train]
        train_y = y_train[index_train]
        test_x = x_train[index_test]
        test_y = y_train[index_test]

        # test_x = np.concatenate((test_x, x_discard), axis=0)
        # test_y = np.concatenate((test_y, y_discard), axis=0)

        eval_set = [(train_x, train_y), (test_x, test_y)]
        model_xgb.fit(train_x, train_y, eval_metric=eval_metric, eval_set=eval_set, early_stopping_rounds=10,
                      verbose=False)

        if metric == "f1":
            y_pred = model_xgb.predict(test_x)
            if len(np.unique(y)) > 2:
                f1 = f1_score(test_y, y_pred, average="macro")
            else:
                f1 = f1_score(test_y, y_pred)
            scores.append(round(f1, 4))
        if metric == "auc":
            y_pred = model_xgb.predict_proba(test_x)
            if len(np.unique(y)) > 2:
                auc = roc_auc_score(test_y, y_pred, average="macro", multi_class="ovo")
            else:
                auc = roc_auc_score(test_y, y_pred[:, 1])
            scores.append(round(auc, 4))
        if metric == "ks":
            y_pred = model_xgb.predict_proba(test_x)[:, 1]
            fpr, tpr, thresholds = roc_curve(test_y, y_pred)
            ks = max(tpr - fpr)
            scores.append(round(ks, 4))

    # 与初始的五折的score对比，reward为均值-下降部分
    values = np.array(scores) - np.array(scores_b)
    mask = values < 0
    negative = values[mask]
    negative_sum = negative.sum()
    reward = np.array(scores).mean() + negative_sum
    return round(reward, 4), scores


def get_reward(x, y, args, scores_b, mode, model, metric):
    x_train = x
    y_train = y
    model = eval(f"{model}_{mode}")()

    if args.cv == 1:
        if mode == "classify":
            my_cv = StratifiedShuffleSplit(n_splits=args.cv, train_size=args.cv_train_size, random_state=args.cv_seed)
        else:
            my_cv = ShuffleSplit(n_splits=args.cv, train_size=args.cv_train_size, random_state=args.cv_seed)
    else:
        if mode == "classify":
            my_cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.cv_seed)
        else:
            my_cv = KFold(n_splits=args.cv, shuffle=True, random_state=args.cv_seed)
    scores = []
    for index_train, index_test in my_cv.split(x_train, y_train):
        train_x = x_train[index_train]
        train_y = y_train[index_train]
        test_x = x_train[index_test]
        test_y = y_train[index_test]

        model.fit(train_x, train_y)
        # print(model.n_iter_)

        if metric == "f1":
            y_pred = model.predict(test_x)
            if len(np.unique(y)) > 2:
                f1 = f1_score(test_y, y_pred, average="micro")
            else:
                f1 = f1_score(test_y, y_pred)
            scores.append(round(f1, 4))
        if metric == "auc":
            y_pred = model.predict_proba(test_x)
            if len(np.unique(y)) > 2:
                auc = roc_auc_score(test_y, y_pred, average="micro", multi_class="ovo")
            else:
                auc = roc_auc_score(test_y, y_pred[:, 1])
            scores.append(round(auc, 4))
        if metric == "ks":
            y_pred = model.predict_proba(test_x)[:, 1]
            fpr, tpr, thresholds = roc_curve(test_y, y_pred)
            ks = max(tpr - fpr)
            scores.append(round(ks, 4))
        if metric == "rae":
            y_pred = model.predict(test_x)
            s = rae(test_y, y_pred)
            scores.append(round(s, 4))

    # scores = cross_val_score(model, x, y, cv=5, scoring=scored)

    # 与初始的五折的score对比，reward为均值-下降部分
    values = np.array(scores) - np.array(scores_b)
    mask = values < 0
    negative = values[mask]
    negative_sum = negative.sum()
    reward = np.array(scores).mean() + negative_sum

    return round(reward, 4), values, scores


def pipe_cal_reward(x, y, actions, model, mode, metric=None):
    x_train = x
    y_train = y
    pipe_args = {'dataframe': pd.concat([x_train, y_train.to_frame()], axis=1),
                 'continuous_columns': x_train.columns,
                 'discrete_columns': [],
                 'label_name': y_train.to_frame().columns[0],
                 'mode': mode,
                 'isvalid': False,
                 'memory': None}
    pipeline = Pipline(pipe_args)
    d_processed, d_ori = pipeline.process_continuous(actions)
    # 原始数据集
    rmodel = eval(f"{model}_{mode}")()
    rmodel.fit(x_train, y_train)
    y_tr_pred = rmodel.predict(x_train)
    accuracy_ori = rmodel.score(x_train, y)
    mse_ori = mean_squared_error(y_train, y_tr_pred)

    # 特征工程后生成的新数据集
    mmodel = eval(f"{model}_{mode}")()
    mmodel.fit(d_processed, y_train)
    y_tr_pro = mmodel.predict(d_processed)
    accuracy_pro = mmodel.score(d_processed, y)
    mse_pro = mean_squared_error(y_train, y_tr_pro)
    # print("---acc---")
    # print(accuracy_pro)
    return d_processed, mse_ori, mse_pro


def ts_split_train_val(df, c):
    x_train = df.iloc[:, 0:(c - 2)]
    y_train = df.iloc[:, (c - 2)]
    x_val = df.iloc[:, 1:(c - 1)]
    y_val = df.iloc[:, (c - 1)]

    return x_train, y_train, x_val, y_val


def binning_c(ori_fe, bins, method="distance"):
    if method == "frequency":
        fre_list = [np.percentile(ori_fe, 100 / bins * i) for i in range(1, bins)]
        fre_list = sorted(list(set(fre_list)))
        new_fe = np.array([map_list(x, fre_list) for x in ori_fe])
        return new_fe
    if method == "distance":
        umax = np.percentile(ori_fe, 99.99)
        umin = np.percentile(ori_fe, 0.01)
        step = (umax - umin) / bins
        fre_list = [umin + i * step for i in range(1, bins)]
        new_fe = np.array([map_list(x, fre_list) for x in ori_fe])
        return new_fe


def map_list(x, fre_list):
    '''
    # 根据数值所在区间，给特征重新赋值，区间左开右闭
    :type x: float, 单个特征的值
    :type fre_list: list of floats,分箱界限
    '''
    if x <= fre_list[0]:
        return 0
    elif x > fre_list[-1]:
        return len(fre_list)
    else:
        for i in range(len(fre_list) - 1):
            if x > fre_list[i] and x <= fre_list[i + 1]:
                return i + 1
