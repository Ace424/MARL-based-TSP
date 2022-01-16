import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from sklearn.metrics import mean_squared_error, mean_absolute_error

import numpy as np
import pandas as pd

BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 100  # After how much time you refresh target network
MEMORY_CAPACITY = 20  # The size of experience replay buffer


class Net(nn.Module):

    def __init__(self, N_STATES, N_ACTIONS):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 100)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization, set seed to ensure the same result
        self.out = nn.Linear(100, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_value = self.out(x)
        return action_value


class DQN(object):
    def __init__(self, N_STATES, N_ACTIONS):
        self.eval_net, self.target_net = Net(N_STATES, N_ACTIONS), Net(N_STATES, N_ACTIONS)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x, N_ACTIONS):
        actions = []
        log_probs = []
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < EPSILON:
            action_value = self.eval_net.forward(x)
            action_softmax = torch.softmax(action_value, dim=-1)
            index_none = []
            for index, out in enumerate(action_softmax):
                dist = Categorical(out)
                if index in index_none:
                    action = torch.tensor(N_ACTIONS - 1)
                else:
                    action = dist.sample()
                log_prob = dist.log_prob(action)
                actions.append(int(action.item()))
                log_probs.append(log_prob.item())
        else:
            action = np.random.randint(0, N_ACTIONS)
            actions.append(action)
        return actions

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a[0], r], s_))
        index = self.memory_counter % MEMORY_CAPACITY  # If full, restart from the beginning
        # print(self.memory[index].shape)
        # print(self.memory[index].dtype)
        self.memory[index] = transition
        self.memory_counter += 1
        # print(self.memory)

    def learn(self, N_STATES):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        # print(b_memory)
        # print(b_memory.dtype)
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1])
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
