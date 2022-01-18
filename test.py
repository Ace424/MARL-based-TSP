import pandas as pd
import logging
from sklearn.metrics import f1_score, r2_score, roc_auc_score, roc_curve
from feature_engineering.pipline_memory_with_selector import generate_fes
from feature_engineering.memory_no_encode import Memory
from model_eval import *
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, ShuffleSplit
from feature_engineering import utils_memory
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_selection import VarianceThreshold
from feature_engineering.feature_select import ModelSelector, chi2_filter, mutual_info_filter
import time
import xgboost as xgb
from config import configs
from scipy.stats import ks_2samp
from collections import Counter
from utils import cluster_data, merge_categories, label_encode_to_onehot
from feature_engineering.feature_generate_memory import *
import shap
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LogisticRegression
from utils import remove_duplication, rae
import latexify


def test_baseline_cv(df: pd.DataFrame, y, args, mode, model, metric):
    # c_columns = configs[args.file_name]["c_columns"]
    # d_columns = configs[args.file_name]["d_columns"]

    # df_process_c = pd.DataFrame()
    # df_process_d = pd.DataFrame()
    # df_t = pd.DataFrame()

    model = eval(f"{model}_{mode}")()

    # 连续特征做归一化，离散特征数值化
    # for column in df.columns.values:
    #     if column in c_columns:
    #         df_process_c[column] = (df[column].values - df[column].values.mean()) / (df[column].values.std())
    #     elif column in d_columns:
    #         df_process_d[column] = utils_memory.categories_to_int(df[column].values)
    #     else:
    #         if mode == "classify":
    #             df_t[column] = utils_memory.categories_to_int(df[column].values)
    #         else:
    #             df_t[column] = df[column].values

    # x_d = label_encode_to_onehot(df_process_d.values)
    # if isinstance(x_d, np.ndarray):
    #     x_train = np.concatenate((df_process_c.values, x_d), axis=1)
    # else:
    #     x_train = df_process_c.values
    # y_train = df_t.values.ravel()
    x_train = df.values
    y_train = y.values.ravel()

    if mode == "classify":
        my_cv = StratifiedShuffleSplit(n_splits=args.cv, train_size=args.cv_train_size, random_state=args.cv_seed)
    else:
        my_cv = ShuffleSplit(n_splits=args.cv, train_size=args.cv_train_size, random_state=args.cv_seed)

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
            if len(np.unique(y_train)) > 2:
                f1 = f1_score(test_y, y_pred, average="micro")
            else:
                f1 = f1_score(test_y, y_pred)
            scores.append(round(f1, 4))
        if metric == "auc":
            y_pred = model.predict_proba(test_x)
            if len(np.unique(y_train)) > 2:
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

    return np.array(scores).mean(), scores
