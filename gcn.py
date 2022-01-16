import numpy as np


# GCN相关的特征表示
def Feature_GCN(X):
    corr_matrix = X.corr().abs()
    corr_matrix[np.isnan(corr_matrix)] = 0
    corr_matrix_ = corr_matrix - np.eye(len(corr_matrix), k=0)
    sum_vec = corr_matrix_.sum()

    for i in range(len(corr_matrix_)):
        corr_matrix_.iloc[:, i] = corr_matrix_.iloc[:, i] / sum_vec[i]
        corr_matrix_.iloc[i, :] = corr_matrix_.iloc[i, :] / sum_vec[i]
    W = corr_matrix_ + np.eye(len(corr_matrix), k=0)
    Feature = np.mean(np.dot(X.values, W.values), axis=1)

    return Feature
