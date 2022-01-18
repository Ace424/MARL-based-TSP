import torch.nn as nn
import torch
import torch.optim as optim
from modules import StatisticLearning, EncoderLayer, SelectOperations, ReductionDimension
from torch.distributions.categorical import Categorical
import os
import pickle
from torch.distributions.bernoulli import Bernoulli
import logging
import random
import numpy as np


class Actor(nn.Module):
    def __init__(self, args, data_nums, operations, d_model, d_k, d_v, d_ff, n_heads, dropout=None):
        super(Actor, self).__init__()
        self.reduction_dimension = ReductionDimension(data_nums, d_model)
        self.encoder1 = EncoderLayer(d_model, d_k, d_v, d_ff, n_heads, dropout)
        self.encoder2 = EncoderLayer(d_model, d_k, d_v, d_ff, n_heads, dropout)
        self.select_operation = SelectOperations(d_model, operations)

    def forward(self, input):
        data_reduction_dimension = self.reduction_dimension(input)
        encoder_output1 = self.encoder1(data_reduction_dimension)
        encoder_output2 = self.encoder2(encoder_output1)
        operation_softmax = self.select_operation(encoder_output2)
        return operation_softmax


class PPO(object):
    def __init__(self, args, data_nums, operations, d_model, d_k, d_v, d_ff, n_heads, dropout=None, c_param=False):
        self.args = args
        self.entropy_weight = args.entropy_weight

        self.epochs = args.epochs
        self.episodes = args.episodes
        self.ppo_epochs = args.ppo_epochs

        # classify or regression
        # self.mode = args.mode

        # 即rf设置中的Agent,负责与环境交互,依据现有状态和策略reward调整action的概率
        self.actor_c = Actor(args, data_nums, operations, d_model, d_k, d_v, d_ff, n_heads, dropout=dropout)

        self.actor_c_opt = optim.Adam(params=self.actor_c.parameters(), lr=args.lr)

        if c_param:
            self.actor_c.load_state_dict(torch.load(c_param)["net"])
            self.actor_c_opt.load_state_dict(torch.load(c_param)["opt"])

        self.baseline = {}
        for step in range(args.steps_num):
            self.baseline[step] = None

        self.baseline_weight = self.args.baseline_weight

        self.clip_epsion = 0.2

    def choose_action_c(self, input_c, step, epoch, ops, sample_rule):
        actions = []
        log_probs = []

        self.actor_c.train()
        action_softmax = self.actor_c(input_c)
        # print("------action_softmax------")
        # print(action_softmax)
        # print(action_softmax.shape)

        index_none = []
        for index, out in enumerate(action_softmax):
            dist = Categorical(out)
            if index in index_none:
                action = torch.tensor(len(ops) - 1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            actions.append(int(action.item()))
            log_probs.append(log_prob.item())
        return actions, log_probs

    def predict_action_c(self, input_c, step):
        actions = []
        # self.actor_c.eval()
        outs = self.actor_c(input_c)
        if step == "selector_c":
            for out in outs:
                if out > 0.5:
                    action = 1
                else:
                    action = 0
                actions.append(action)
        else:
            for out in outs:
                action = out.argmax()
                actions.append(action)
        return actions

    def update_c(self, workers_c):
        # 从worker中取出state,reward,action等
        rewards = []
        dones = []
        for worker in workers_c:
            rewards.extend(worker.accs)
            dones.extend(worker.dones)
        # 计算长期的reward

        rewards_convert = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.args.gama * discounted_reward)
            rewards_convert.insert(0, discounted_reward)
        # 计算每个step的baseline，滑动平均
        for step in range(self.args.steps_num):
            for reward in rewards_convert[step::self.args.steps_num]:
                if self.baseline[step] == None:
                    self.baseline[step] = reward
                else:
                    self.baseline[step] = self.baseline[step] * self.baseline_weight + reward * (
                            1 - self.baseline_weight)
        baseline_step = []
        for step in range(self.args.steps_num):
            baseline_step.append(self.baseline[step])

        baseline_step = torch.tensor(baseline_step)
        rewards_convert = torch.tensor(rewards_convert).reshape(-1, self.args.steps_num)
        advantages = rewards_convert - baseline_step

        for epoch in range(self.args.ppo_epochs):
            total_loss = 0
            total_loss_actor = 0
            total_loss_entorpy = 0
            for worker_index, worker in enumerate(workers_c):
                old_log_probs_ = worker.log_probs
                states = worker.states
                actions = worker.actions
                steps = worker.steps

                advantage = advantages[worker_index]
                advantage_convert = []

                for i, log_pros in enumerate(old_log_probs_):
                    advantage_ = advantage[i]
                    for j, log_pro in enumerate(log_pros):
                        advantage_convert.append(advantage_)
                advantage_convert = torch.tensor(advantage_convert)

                # advantage和log_probs转成一维
                old_log_probs = torch.tensor([item for x in old_log_probs_ for item in x])

                new_log_probs = []
                entropys = []
                for index, state in enumerate(states):
                    action = actions[index]
                    step = steps[index]
                    action_softmax = self.actor_c(state)
                    if index == 0:
                        softmax_output = action_softmax
                    for k, out in enumerate(action_softmax):
                        dist = Categorical(out)
                        entropy = dist.entropy()
                        entropys.append(entropy.unsqueeze(dim=0))
                        new_log_prob = dist.log_prob(torch.tensor(action[k])).unsqueeze(dim=0).float()
                        new_log_probs.append(new_log_prob)

                new_log_probs = torch.cat(new_log_probs)
                entropys = torch.cat(entropys)

                # ppo公式
                prob_ratio = new_log_probs.exp() / old_log_probs.exp()
                weighted_probs = advantage_convert * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.clip_epsion,
                                                     1 + self.clip_epsion) * advantage_convert
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs)
                actor_loss = actor_loss.sum()

                entropy_loss = entropys.sum()
                total_loss_actor += actor_loss
                total_loss_entorpy += (- self.args.entropy_weight * entropy_loss)
                total_loss += (actor_loss - self.args.entropy_weight * entropy_loss)
            total_loss /= len(workers_c)
            actor_loss = total_loss_actor / len(workers_c)
            entropy_loss = total_loss_entorpy / len(workers_c)

            value, index = softmax_output.max(dim=-1)
            logging.info(f"value1:{value},index1:{index}")
            value, index = action_softmax.max(dim=-1)
            logging.info(f"value2:{value},index2:{index}")
            logging.info(
                f"total_loss:{total_loss.item()},actor_loss:{actor_loss.item()},entory_loss:{entropy_loss.item()}")
            self.actor_c_opt.zero_grad()
            total_loss.backward()
            self.actor_c_opt.step()

    def save_model_c(self):
        dir = f"./params/dl"
        name = self.args.file_name.split(".")[0]
        if not os.path.exists(dir):
            os.makedirs(dir)
        torch.save({"net": self.actor_c.state_dict(), "opt": self.actor_c_opt.state_dict()},
                   f"{dir}/{name}_actor_c.pth")


if __name__ == '__main__':
    # actor = Actor(n_features=12, statistic_nums=5, operations=6, n_value_converts=8, d_model=64, d_k=32, d_v=32,
    #               d_ff=128, n_heads=6)
    # x = torch.randn(1, 6, 5)
    # y = actor(x)
    # print(y[0])
    # print(y[2])
    # print(y[2].shape)

    # flops,params = profile(actor,inputs=(x,))
    # print("actor:",flops,params)

    x = torch.tensor([1, 3, 4])
    value, index = x.max()
