from ppo import PPO
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import logging
import torch
from parsers import parse_actions
from utils import features_process, feature_merge, split_train_test, get_reward, pipe_cal_reward, ts_split_train_val, \
    merge_categories, \
    label_encode_to_onehot, calculate_statistics, remove_duplication, get_ops
from feature_engineering.pipline_memory_with_selector import Pipline
from feature_engineering.feature_generate_memory import binning_with_tree
from worker import Worker
from test import test_baseline_cv

from gcn import Feature_GCN
from dqn import Net, DQN

from model_eval import *
from multiprocessing import Process, Queue, Manager, Pool
from sklearn.feature_selection import VarianceThreshold

from math import log


def getLinerFlag(dataSet, axis):
    num = len(dataSet)
    for i in range(num):
        feature = [example[axis] for example in dataSet]
        feature2 = sorted(feature)
    flag = feature[int(num / 2)]
    return flag


def calcLinerData(dataSet):
    num = len(dataSet)
    count = {1: 0, 0: 0}
    shannonEnt = 0.0
    feature = []
    feature2 = []
    for i in range(num):
        feature = [example[-1] for example in dataSet]
        feature2 = sorted(feature)
    flag = feature2[int(num / 2)]
    for i in range(num):
        if feature[i] >= flag:
            feature[i] = 1
            count[1] += 1
        else:
            feature[i] = 0
            count[0] += 1
    for i in [0, 1]:
        prob = float(count[i]) / num
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)  # 分母：训练数据的数量
    labelCounts = {}  # 分子：数据集或者每个子集中，每个类别（好瓜、坏瓜）出现的次数
    # 给所有可能分类创建字典
    for featVec in dataSet:
        currentLabel = featVec[-1]  # 取最后一列数据
        if currentLabel not in labelCounts.keys():  # 第一次出现时先给它初始值0
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    # 以2为底数计算香农熵
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)  # Ent=-(∑pk㏒pk)  --> Ent减每一个结果 P75(4.1)
    return shannonEnt


def splitDataSet(dataSet, axis, value, direction):
    retDataSet = []
    for featVec in dataSet:
        if direction == 0:
            if featVec[axis] > value:
                retDataSet.append(featVec)  # 连续型特征和特征值都不删除
        else:
            if featVec[axis] <= value:
                retDataSet.append(featVec)
    return retDataSet


# trained by PPO
def trainPPO(args, dataset_path, target, covariates, mode, model, metric):
    df_target = pd.read_csv(dataset_path + target)
    df_covariates = [pd.read_csv(dataset_path + x) for x in covariates]
    r, c = df_target.shape

    # 划分训练和测试集
    X_train, Y_train, X_val, Y_val = ts_split_train_val(df_target, c)
    n_features_c = X_train.shape[1]  # feature number
    ops = get_ops(n_features_c)  # feature ops

    # 训练集五折验证作为baseline，返回均值及五折的值
    score_b, scores_b = test_baseline_cv(X_train, Y_train, args, mode, model, metric)

    data_nums = df_target.shape[0]
    operations = len(ops)
    covars = len(covariates)
    d_model = args.d_model
    d_k = args.d_k
    d_v = args.d_v
    d_ff = args.d_ff
    n_heads = args.n_heads
    ppo = PPO(args, data_nums, operations, d_model, d_k, d_v, d_ff, n_heads, dropout=None)

    pipeline_args_train = {'dataframe': df_target,
                           'continuous_columns': X_train.columns,
                           'discrete_columns': [],
                           'label_name': Y_train.to_frame().columns[0],
                           'mode': mode,
                           'isvalid': False,
                           'memory': None}

    # 用于存储协变量序列的Agent
    ppo_list = []
    for agent in range(len(covariates)):
        data_co_nums = df_covariates[agent].shape[0]
        ppo_list.append(PPO(args, data_co_nums, operations, d_model, d_k, d_v, d_ff, n_heads, dropout=None))

    workers_c = []
    co_workers = []
    best_worker_target = Worker(args)  # 记录最优target worker
    best_mixed_target = Worker(args)  # 记录最优mixed worker
    best_coworkers = [Worker(args) for i in range(covars)]  # 记录最优co workers

    for epoch in range(args.epochs):
        queue = Queue()
        df_mixed = [[] for i in range(args.steps_num)]
        result_mix = Worker(args)
        # sample步骤进行Action选取和State变更
        # target序列采样
        result_target = sample(args, ppo, pipeline_args_train, pd.concat([X_train, Y_train.to_frame()], axis=1),
                               Y_train, ops, epoch)
        workers_c.append(result_target)
        # target序列use cooperative features

        # 协变量序列采样
        for agent in range(covars):
            df_co = df_covariates[agent]
            co = df_co.shape[1]
            X_train_co, Y_train_co, X_val_co, Y_val_co = ts_split_train_val(df_co, co)
            # try co-label == target
            pipeline_args_co = {'dataframe': pd.concat([X_train_co, Y_train], axis=1),
                                'continuous_columns': X_train_co.columns,
                                'discrete_columns': [],
                                'label_name': Y_train_co.to_frame().columns[0],
                                'mode': mode,
                                'isvalid': False,
                                'memory': None}
            result_co = sample(args, ppo_list[agent], pipeline_args_co,
                               pd.concat([X_train_co, Y_train_co.to_frame()], axis=1), Y_train_co, ops, epoch)
            co_workers.append(result_co)

        # 各Agent计算reward&update
        # target序列计算reward
        accs = []
        cvs = []
        for step in range(args.steps_num):
            ff = result_target.ff[0:step + 1]
            x = result_target.features[step]
            df_mixed[step] = x
            y = Y_train.values
            acc, cv, _ = get_reward(x, y, args, scores_b, mode, model, metric)
            accs.append(acc)
            cvs.append(cv)
            # 更改acc，cv赋值
        result_target.accs = accs
        result_target.cvs = cvs
        # 记录最优target_worker
        if np.mean(result_target.accs) > np.mean(best_worker_target.accs):
            best_worker_target = result_target

        new_nums = cal_feaure_nums(result_target.ff)
        ori_nums = df_target.shape[1] - 1
        feature_nums = ori_nums + new_nums
        logging.info(
            f"target ,results:{result_target.accs},cv:{result_target.cvs[-1]},feature_nums:{feature_nums / ori_nums, feature_nums, ori_nums},ff:{result_target.ff}")

        # 协变量序列计算reward
        for agent in range(covars):
            accs_co = []
            cvs_co = []
            df_co = df_covariates[agent]
            c = df_co.shape[1]
            X_train_co, Y_train_co, X_val_co, Y_val_co = ts_split_train_val(df_co, c)
            score_b_co, scores_b_co = test_baseline_cv(X_train_co, Y_train_co, args, mode, model, metric)
            # 计算每个step的reward以及accuracy
            for step in range(args.steps_num):
                ff = co_workers[agent].ff[0:step + 1]
                x = co_workers[agent].features[step]
                df_mixed[step] = np.concatenate([df_mixed[step], x], axis=1)
                y = Y_train_co.values
                acc, cv, _ = get_reward(x, y, args, scores_b_co, mode, model, metric)
                accs_co.append(acc)
                cvs_co.append(cv)
            # 更改acc，cv赋值
            co_workers[agent].accs = accs_co
            co_workers[agent].cvs = cvs_co
            # 记录最优co-worker
            if np.mean(accs_co) > np.mean(best_coworkers[agent].accs):
                best_coworkers[agent] = co_workers[agent]

            new_nums = cal_feaure_nums(co_workers[agent].ff)
            ori_nums = df_co.shape[1] - 1
            feature_nums = ori_nums + new_nums
            logging.info(
                f"agent{agent + 1} ,results:{co_workers[agent].accs},cv:{co_workers[agent].cvs[-1]},feature_nums:{feature_nums / ori_nums, feature_nums, ori_nums},ff:{co_workers[agent].ff}")

        # list各epoch's target worker的准确率
        baseline = [worker.accs for worker in workers_c]
        logging.info(f"epoch:{epoch},baseline:{baseline},score_b:{score_b},scores_b:{scores_b}")

        # 计算mixed feature set的准确率
        accs_mix = []
        cvs_mix = []
        for step in range(args.steps_num):
            x = df_mixed[step]
            y = Y_train.values
            acc, cv, _ = get_reward(x, y, args, scores_b, mode, model, metric)
            accs_mix.append(acc)
            cvs_mix.append(cv)
        result_mix.accs = accs_mix
        result_mix.cvs = cvs_mix
        if np.mean(result_mix.accs) > np.mean(best_mixed_target.accs):
            best_mixed_target = result_mix

        mixed_nums = df_mixed[args.steps_num - 1].shape[1]
        ori_nums = df_target.shape[1] - 1
        logging.info(
            f"mixed-cotarget ,results:{result_mix.accs},cv:{result_mix.cvs[-1]},feature_nums:{mixed_nums / ori_nums, mixed_nums, ori_nums}")
        # list当前最优target worker, mixed worker的信息
        try:
            new_nums = cal_feaure_nums(best_worker_target.ff)
            ori_nums = df_target.shape[1] - 1
            feature_nums = ori_nums + new_nums
            logging.info(
                f"top_target_acc:{best_worker_target.accs},feature_nums:{feature_nums / ori_nums, feature_nums, ori_nums},{best_worker_target.ff}")

            mixed_nums = df_mixed[args.steps_num - 1].shape[1]
            logging.info(
                f"top_mixed_cotarget_acc:{best_mixed_target.accs},cv:{best_mixed_target.cvs[-1]},feature_nums:{mixed_nums / feature_nums, mixed_nums, feature_nums}")
        except:
            pass

        # ppo更新网络
        ppo.update_c([result_target])
        for agent in range(covars):
            result_co = co_workers[agent]
            ppo_list[agent].update_c([result_co])


def sample(args, ppo, pipline_args_train, df_c_encode, df_t, ops, epoch):
    # 记录采样过程中动作，概率，state，特征等，返回一个worker
    pipline_ff_c = Pipline(pipline_args_train)
    worker_c = Worker(args)
    states_c = []
    actions_acs = []
    log_probs_acs = []
    steps = []
    features_c = []

    # 连续和离散特征初始的state
    n_features = df_c_encode.shape[1] - 1
    # print("------base entropy------")
    # base_entropy = calcShannonEnt(df_c_encode)
    # print(base_entropy)
    init_state_c = torch.from_numpy(df_c_encode.values).float().transpose(0, 1)
    init_feature = df_c_encode.values

    # 连续特征和离散特征的处理步骤
    steps_num = args.steps_num
    # if i < args.episodes // 2:
    sample_rule = True
    # else:
    #     sample_rule = False

    ff = []
    for step_c in range(steps_num):
        """连续采样"""
        state_c = init_state_c
        new_feature = init_feature

        actions, log_probs = ppo.choose_action_c(state_c, step_c, epoch, ops, sample_rule)
        ff_c = parse_actions(actions, ops, n_features)
        ff.append(ff_c)
        steps.append(step_c)
        # 根据每列特征的操作生成新的特征，更新state
        for ff_action in ff_c:
            x_c_norm, x_c = pipline_ff_c.process_continuous(ff_action)
        x_c_norm, idx = remove_duplication(x_c_norm)

        var_selector = VarianceThreshold()
        x_c_norm = var_selector.fit_transform(x_c_norm)

        features_c.append(x_c_norm)

        x_encode_c = np.hstack((x_c_norm, df_t.values.reshape(-1, 1)))
        init_feature = x_encode_c
        x_encode_c = torch.from_numpy(x_encode_c).float().transpose(0, 1)
        init_state_c = x_encode_c

        states_c.append(state_c)
        actions_acs.append(actions)
        log_probs_acs.append(log_probs)

    # 采样是否完成，分三个step即[False,False,True]
    dones = [False for i in range(steps_num)]
    dones[-1] = True

    worker_c.steps = steps
    worker_c.states = states_c
    worker_c.actions = actions_acs
    worker_c.log_probs = log_probs_acs
    worker_c.dones = dones
    worker_c.features = features_c
    worker_c.ff = ff
    # queue.put(worker_c)
    return worker_c


def cal_feaure_nums(ff):
    nums = 0
    for f in ff:
        for ops in f:
            if list(ops.keys())[0] == "value_convert":
                ops_value = ops["value_convert"].values()
                none_nums = list(ops_value).count("None")
                num = len(ops_value) - none_nums
                nums += num
            elif list(ops.keys())[0] == "delete":
                ops_key = ops["delete"].keys()
                delete_nums = len(list(ops_key))
                nums -= delete_nums
            else:
                num = len(list(ops.values())[0])
                nums += num
    return nums


# trained by DQN
BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 100  # After how much time you refresh target network
MEMORY_CAPACITY = 3  # The size of experience replay buffer


def trainDQN(args, dataset_path, target, covariates, mode, model, metric):
    df_target = pd.read_csv(dataset_path + target)
    df_covariates = [pd.read_csv(dataset_path + x) for x in covariates]
    r, c = df_target.shape

    X_train, Y_train, X_val, Y_val = ts_split_train_val(df_target, c)

    N_feature = X_train.shape[1]  # feature number
    ops = get_ops(N_feature)  # feature ops
    N_sample = X_train.shape[0]  # feature length,i.e., sample number
    N_ACTIONS = len(ops)
    N_STATES = len(X_train)

    print("N_ACTIONS:" + str(N_ACTIONS))

    np.random.seed(0)
    dqn = DQN(N_STATES=N_STATES, N_ACTIONS=N_ACTIONS)

    d_target = X_train.values  # dataset of target
    s_target = Feature_GCN(X_train)
    # s_target = torch.from_numpy(X_train.values).float().transpose(0, 1)
    d_covariates = []  # datasets of covariates
    s_covariates = []

    np.random.seed(0)
    dqn_list = []
    result = []
    for agent in range(len(covariates)):
        dqn_list.append(DQN(N_STATES=N_STATES, N_ACTIONS=N_ACTIONS))
        d_covariates.append(df_covariates[agent].values)
        s_covariates.append(Feature_GCN(df_covariates[agent]))
        # s_covariates.append(torch.from_numpy(df_covariates[agent].values).float().transpose(0, 1))

    # 记录各Agent的reward
    acc_ori = 0
    acc_co_oris = []
    reward_target_pros = []
    reward_target_vals = []
    reward_mix_pros = []
    reward_mix_vals = []
    reward_co_pros = [[] for i in range(len(covariates))]
    reward_co_vals = [[] for i in range(len(covariates))]
    action_list = [[] for i in range(len(covariates))]

    for i in range(args.epochs):
        # 各Agent选择自己的动作
        target_action = dqn.choose_action(s_target, N_ACTIONS)
        # print("-----target action------")
        # print(target_action)

        for agent, dqn in enumerate(dqn_list):
            action_list[agent] = dqn.choose_action(s_covariates[agent], N_ACTIONS)
            # print("-----agent " + str(agent + 1) + "------")
            # print(action_list[agent])

        # 各Agent需要重新构造特征集合d_...,计算下一个状态s_...
        target_parse_action = parse_actions(target_action, ops, N_feature)
        logging.info(f"epoch:{i + 1} ,target_actions:{target_parse_action}")
        d_target_next, reward_target_pro, reward_target_ori = pipe_cal_reward(X_train, Y_train, target_parse_action,
                                                                              model, mode)
        d_target_val, reward_target_val, reward_val_ori = pipe_cal_reward(X_val, Y_val, target_parse_action, model,
                                                                          mode)
        d_target_mix = d_target_next
        d_val_mix = d_target_val
        acc_ori = reward_target_ori
        reward_target_pros.append(reward_target_pro)
        reward_target_vals.append(reward_target_val)

        s_target_next = Feature_GCN(pd.DataFrame(d_target_next))
        # s_target_next = torch.from_numpy(d_target_next).float().transpose(0, 1)
        dqn.store_transition(s_target, target_action, reward_target_pro, s_target_next)
        # s_target = s_target_next
        acc_co_oris = []
        for agent, dqn in enumerate(dqn_list):
            c = df_covariates[agent].shape[1]
            X_cotrain, Y_cotrain, X_coval, Y_coval = ts_split_train_val(df_covariates[agent], c)
            co_parse_action = parse_actions(action_list[agent], ops, N_feature)
            logging.info(f"agent:{agent + 1} ,co_actions:{co_parse_action}")
            d_co_next, reward_co_pro, reward_co_ori = pipe_cal_reward(X_cotrain, Y_cotrain, co_parse_action, model,
                                                                      mode)
            d_co_val, reward_co_val, reward_cval_ori = pipe_cal_reward(X_coval, Y_coval, co_parse_action, model, mode)

            d_target_mix = np.concatenate((d_target_mix, d_co_next), axis=1)
            d_val_mix = np.concatenate((d_val_mix, d_co_val), axis=1)

            acc_co_oris.append(reward_co_ori)
            reward_co_pros[agent].append(reward_co_pro)
            reward_co_vals[agent].append(reward_co_val)

            s_covar_next = Feature_GCN(pd.DataFrame(d_co_next))
            # s_covar_next = torch.from_numpy(d_co_next).float().transpose(0, 1)
            dqn.store_transition(s_covariates[agent], action_list[agent], reward_co_pro, s_covar_next)
            # s_covariates[agent] = s_covar_next

        # 新特征组合
        rfmodel = eval(f"{model}_{mode}")()
        rfmodel.fit(d_target_mix, Y_train)
        accuracy_mix_pro = rfmodel.score(d_target_mix, Y_train)
        reward_mix_pros.append(accuracy_mix_pro)
        rfmodel.fit(d_val_mix, Y_val)
        accuracy_mix_val = rfmodel.score(d_val_mix, Y_val)
        reward_mix_vals.append(accuracy_mix_val)

        # dqn经验重放算法
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn(N_STATES)
        if dqn_list[0].memory_counter > MEMORY_CAPACITY:
            for dqn in dqn_list:
                dqn.learn(N_STATES)

    print("initial reward_target: " + str(reward_target_ori))

    plt.title('accuracy per epoch')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.plot(range(args.epochs), reward_target_pros, label='target_acc_pro')
    plt.plot(range(args.epochs), reward_target_vals, label='target_acc_val')
    plt.plot(range(args.epochs), reward_mix_pros, label='mixed_acc_pro')
    plt.plot(range(args.epochs), reward_mix_vals, label='mixed_acc_val')
    for agent in range(len(covariates)):
        plt.plot(range(args.epochs), reward_co_pros[agent], label='covar_' + str(agent + 1) + '_acc_pro')
        plt.plot(range(args.epochs), reward_co_vals[agent], label='covar_' + str(agent + 1) + '_acc_val')
    plt.legend()
    plt.savefig('test_deep_layer.png')
    plt.show()
