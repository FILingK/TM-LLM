import numpy as np
import torch

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def NRMSE_mean(pred, true):
    # 计算 RMSE
    rmse = np.sqrt(np.sum((pred - true) ** 2))
    # 计算真实值的均值
    mean_true = np.sqrt(np.sum(true**2))
    print("rmse: {0}, mean_true{1}".format(rmse,mean_true))
    # 返回归一化后的 RMSE，防止除以零
    return rmse / mean_true if mean_true != 0 else rmse

def NMAE_mean(pred, true):
    mae = np.sum(np.abs(pred - true))
    mean_true = np.sum(np.abs(true))  # 计算真实值的平均值
    print("mae: {0}, mean_true{1}".format(mae, mean_true))
    return mae / mean_true if mean_true != 0 else mae  # 防止除以零

def compute_dcg_at_k(ranking, k):
    """
    计算给定排序的 DCG@k
    :param ranking: 排序的流量值
    :param k: K值，取前K个
    :return: DCG@k
    """
    dcg = 0
    for i in range(k):
        dcg += (2 ** ranking[i] - 1) / np.log2(i + 2)
    return dcg


# def ndcg(preds, trues, masks, k=10):
#     """
#     计算 NDCG@k
#     :param preds: 预测结果，形状为 (B, T, N)
#     :param trues: 真实值，形状为 (B, T, N)
#     :param masks: 掩码矩阵，形状为 (B, T, N)
#     :param k: NDCG@k 中的 k 值，默认是 10
#     :return: 每个时间步的平均 NDCG@k
#     """
#     B, T, N = preds.shape
#     ndcg_list = []
#
#     # 对每一个时间步计算 NDCG@k
#     for t in range(T):
#         # 获取每个时间步的预测和真实流量
#         pred_t = preds[:, t, :]  # 预测值，大小为 (B, N)
#         true_t = trues[:, t, :]  # 真实值，大小为 (B, N)
#         mask_t = masks[:, t, :]  # 掩码矩阵，大小为 (B, N)
#
#         dcg_t = 0
#         idcg_t = 0
#
#         # 对每个 batch 计算 DCG 和 IDCG
#         for b in range(B):
#             # 仅使用有效的（masks == 0）数据
#             valid_pred = pred_t[b][mask_t[b] == 0]  # 预测值，只保留有效数据
#             valid_true = true_t[b][mask_t[b] == 0]  # 真实值，只保留有效数据
#
#             valid_pred = torch.tensor(valid_pred, dtype=torch.float32)
#             valid_true = torch.tensor(valid_true, dtype=torch.float32)
#             # 对有效的预测和真实流量进行排序，按降序排列
#             pred_sorted = torch.argsort(valid_pred, descending=True)
#             true_sorted = torch.argsort(valid_true, descending=True)
#
#             # 获取前k个元素的 DCG 和 IDCG
#             dcg_t += compute_dcg_at_k(valid_pred[pred_sorted], k)
#             idcg_t += compute_dcg_at_k(valid_true[true_sorted], k)
#
#         # 计算当前时间步的 NDCG@k
#         ndcg_t = dcg_t / idcg_t if idcg_t != 0 else 0
#         ndcg_list.append(ndcg_t)
#
#     # 计算所有时间步的平均 NDCG@k
#     mean_ndcg = np.mean(ndcg_list)
#     return mean_ndcg
#

def get_rank_dict(arr, max_rank=530):
    """
    获取数组的排名（降序排名），并转换为 `max_rank - i`
    :param arr: 输入数组 (Tensor)
    :param max_rank: 最高排名，默认 144
    :return: 相关性排名数组（保持原顺序）
    """
    arr_np = arr.numpy()
    sorted_indices = np.argsort(arr_np)[::-1]  # 降序排序索引
    ranks = {arr_np[i]: max_rank - rank for rank, i in enumerate(sorted_indices)}  # 计算排名
    return np.array([ranks[val] for val in arr_np])  # 返回排名数组，保持原顺序

# ndcg@10
def find_top_k_ranks(true_ranked, pred_ranked, k=10):
    """
    找到 `true_ranked` 前 k 名的索引，并获取这些索引在 `pred_ranked` 和 `true_ranked` 中的排名
    :param true_ranked: 真实值的排名数组（保持原顺序）
    :param pred_ranked: 预测值的排名数组（保持原顺序）
    :param k: 取前 k 个
    :return: (top_k_indices, pred_ranks_at_top_k, true_ranks_at_top_k)
    """
    # 1. 找到 `true_ranked` 前 K 名的索引
    top_k_indices = np.argsort(true_ranked)[-k:]  # 获取前 k 名的索引

    # 2. 在 `pred_ranked` 和 `true_ranked` 中找到这些索引对应的排名
    pred_ranks_at_top_k = pred_ranked[top_k_indices]
    true_ranks_at_top_k = true_ranked[top_k_indices]

    return top_k_indices, pred_ranks_at_top_k, true_ranks_at_top_k

def compute_dcg_at_k(ranking, k):
    """
    计算给定排序的 DCG@k
    :param ranking: 排序的流量值
    :param k: K值，取前K个
    :return: DCG@k
    """
    dcg = 0
    for i in range(k):
        # 除以 10 来避免数值过大
        dcg += (2 ** (ranking[i]/30.0) - 1) / np.log2(i + 2)
    return dcg


def ndcg(preds, trues, masks, k=10, max_rank=530):
    """
    计算 NDCG@k，同时记录 `true_ranked` 前 K 的索引、`pred_ranked` 和 `true_ranked` 的排名
    """
    B, T, N = preds.shape
    ndcg_list = []

    for t in range(T):
        pred_t = preds[:, t, :]
        true_t = trues[:, t, :]
        mask_t = masks[:, t, :]

        dcg_t = 0
        idcg_t = 0

        for b in range(B):
            valid_pred = pred_t[b][mask_t[b] == 0]
            valid_true = true_t[b][mask_t[b] == 0]

            if len(valid_pred) == 0:
                continue

            valid_pred = torch.tensor(valid_pred, dtype=torch.float32)
            valid_true = torch.tensor(valid_true, dtype=torch.float32)

            # **获取排名，但采用 `530 - rank` 进行变换**
            pred_ranked = get_rank_dict(valid_pred, max_rank)
            true_ranked = get_rank_dict(valid_true, max_rank)

            # 找到 `true_ranked` 前 K 的索引及 `pred_ranked` 和 `true_ranked` 中的排名
            top_k_indices, pred_ranks_at_top_k, true_ranks_at_top_k = find_top_k_ranks(true_ranked, pred_ranked, k)

            # 计算 DCG 和 IDCG
            dcg_t += compute_dcg_at_k(pred_ranks_at_top_k, k)
            idcg_t += compute_dcg_at_k(true_ranks_at_top_k, k)  # 直接使用预测数据对应的真实排名

        ndcg_t = dcg_t / idcg_t if idcg_t != 0 else 0
        ndcg_list.append(ndcg_t)

    return np.mean(ndcg_list)

#kl
def normalize_to_probabilities(arr):
    """
    将数组归一化为概率分布
    """
    arr = np.asarray(arr)
    sum_arr = np.sum(arr)
    if sum_arr == 0:
        return arr  # 如果总和为零，直接返回
    return arr / sum_arr
def KL(p, q):
    """
    计算 Kullback-Leibler Divergence (KL Divergence) 适用于预测值和真实值
    """
    epsilon = 1e-10
    p = np.asarray(p)
    q = np.asarray(q)

    # 归一化为概率分布
    p = normalize_to_probabilities(p)
    q = normalize_to_probabilities(q)

    # 防止 log(0) 导致问题
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)

    # 计算 KL Divergence
    output = np.sum(p * np.log(p / q))

    return output


def metric(pred, true):
    nmae = NMAE_mean(pred, true)
    nrmse = NRMSE_mean(pred, true)
    kl_v = KL(true, pred)
    mspe = MSPE(pred, true)

    return nmae, nrmse, kl_v, mspe
