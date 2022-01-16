import numpy as np
import torch


def parse_actions(actions, ops, n_features):
    add = []
    subtract = []
    multiply = []
    divide = []
    diff = []
    value_convert = {}
    delete = {}
    for index, action in enumerate(actions):
        if 0 <= action < n_features:
            # add
            if index < n_features:
                if index != action:
                    add.append([index, action])
            else:
                add.append([index, action])
        elif n_features <= action < (2 * n_features):
            # subtract
            if index < n_features:
                if index != (action - n_features):
                    subtract.append([index, action - n_features])
            else:
                subtract.append([index, action - n_features])

        elif (2 * n_features) <= action < (3 * n_features):
            # multiply
            if index < n_features:
                if index != (action - n_features * 2):
                    multiply.append([index, action - n_features * 2])
            else:
                multiply.append([index, action - n_features * 2])
        elif (3 * n_features) <= action < (4 * n_features):
            # divide
            if index < n_features:
                if index != (action - n_features * 3):
                    divide.append([index, action - n_features * 3])
            else:
                divide.append([index, action - n_features * 3])
        elif (4 * n_features) <= action < (5 * n_features):
            # diff
            if index < n_features:
                if index != (action - n_features * 4):
                    diff.append([index, action - n_features * 4])
            else:
                diff.append([index, action - n_features * 4])
        elif ops[action] == "delete":
            delete[index] = ops[action]
        else:
            value_convert[index] = ops[action]
    action_all = [{"add": add}, {"subtract": subtract}, {"multiply": multiply}, {"divide": divide}, {"diff": diff},
                  {"value_convert": value_convert}, {"delete": delete}]

    return action_all


if __name__ == '__main__':
    index = torch.tensor([1, 0, 1, 0, 1, 1, 0, 1, 0, 1])
    dic = parse_actions(index, "selector")
    print(dic)
