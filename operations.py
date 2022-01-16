# TODO: 增加时序相关的操作
OPS = {
    "arithmetic": ["add", "subtract", "multiply", "divide", "diff"],
    "value_convert": ["log", "delete"]
}
if __name__ == '__main__':
    ops = []
    n_features = 6
    arithmetic = ["add", "subtract", "multiply", "divide", "diff"]
    value_convert = ["log", "delete"]
    for i in range(1, 6):
        op = arithmetic[i - 1]
        for j in range((i - 1) * n_features, i * n_features):
            ops.append(op)
    ops.extend(value_convert)
    ops.append("None")
    print(ops)
    print(len(ops))
