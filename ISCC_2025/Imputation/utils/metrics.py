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
    rmse = np.sqrt(np.sum((pred - true) ** 2))
    mean_true = np.sqrt(np.sum(true ** 2))
    return rmse / mean_true if mean_true != 0 else rmse


def NMAE_mean(pred, true):
    mae = np.sum(np.abs(pred - true))
    mean_true = np.sum(np.abs(true))
    return mae / mean_true if mean_true != 0 else mae


# ndcg@10
def get_rank_dict(arr, max_rank=530):
    """
    Get the rankings of an array (in descending order) and transform them to `max_rank - i`
    For the Abilene dataset, max_rank is set to 145 ; for the Geant dataset, max_rank is set to 530 (which equals the number of features + 1).
    """
    arr_np = arr.numpy()
    sorted_indices = np.argsort(arr_np)[::-1]  # Sort indices in descending order
    ranks = {arr_np[i]: max_rank - rank for rank, i in enumerate(sorted_indices)}  # Compute the rankings
    return np.array([ranks[val] for val in arr_np])  # Return the ranking array, preserving the original order


def find_top_k_ranks(true_ranked, pred_ranked, k=10):
    """
    Find the top k indices of `true_ranked` and retrieve their ranks in `pred_ranked` and `true_ranked`
    :param true_ranked: The ranking array of true values (preserving the original order)
    :param pred_ranked: The ranking array of predicted values (preserving the original order)
    :param k: Take the top k items
    :return: (top_k_indices, pred_ranks_at_top_k, true_ranks_at_top_k)
    """
    # 1. Find the top k indices in `true_ranked`
    top_k_indices = np.argsort(true_ranked)[-k:]  # Get indices of the top k items

    # 2. Retrieve the corresponding ranks of these indices in `pred_ranked` and `true_ranked`
    pred_ranks_at_top_k = pred_ranked[top_k_indices]
    true_ranks_at_top_k = true_ranked[top_k_indices]

    return top_k_indices, pred_ranks_at_top_k, true_ranks_at_top_k


def compute_dcg_at_k(ranking, k):
    """
    Calculate DCG@k for a given ranking
    :param ranking: Ranked traffic values
    :param k: The value of k, considering the top k items
    :return: DCG@k
    """
    dcg = 0
    for i in range(k):
        # Divide by 30 to avoid large numerical values
        dcg += (2 ** (ranking[i] / 30.0) - 1) / np.log2(i + 2)
    return dcg


def ndcg(preds, trues, masks, features, k=10):
    B, T, N = preds.shape
    ndcg_list = []
    max_rank = features + 1
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

            # **Obtain rankings, but use `530 - rank` transformation**
            pred_ranked = get_rank_dict(valid_pred, max_rank)
            true_ranked = get_rank_dict(valid_true, max_rank)

            # Find the top K indices of `true_ranked` and their ranks in `pred_ranked` and `true_ranked`
            top_k_indices, pred_ranks_at_top_k, true_ranks_at_top_k = find_top_k_ranks(true_ranked, pred_ranked, k)

            dcg_t += compute_dcg_at_k(pred_ranks_at_top_k, k)
            idcg_t += compute_dcg_at_k(true_ranks_at_top_k,
                                       k)  # Directly use the true ranks corresponding to predicted data

        ndcg_t = dcg_t / idcg_t if idcg_t != 0 else 0
        ndcg_list.append(ndcg_t)

    return np.mean(ndcg_list)


# kl
def normalize_to_probabilities(arr):
    """
    Normalize the array to a probability distribution
    """
    arr = np.asarray(arr)
    sum_arr = np.sum(arr)
    if sum_arr == 0:
        return arr  # If the sum is zero, return the array as is
    return arr / sum_arr


def KL(p, q):
    """
    Calculate Kullback-Leibler Divergence (KL Divergence) for predicted values and true values
    """
    epsilon = 1e-10
    p = np.asarray(p)
    q = np.asarray(q)

    # Normalize to probability distributions
    p = normalize_to_probabilities(p)
    q = normalize_to_probabilities(q)

    # Prevent issues from log(0)
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)

    # Compute KL Divergence
    output = np.sum(p * np.log(p / q))

    return output


def metric(pred, true):
    nmae = NMAE_mean(pred, true)
    nrmse = NRMSE_mean(pred, true)
    kl_v = KL(true, pred)
    mspe = MSPE(pred, true)

    return nmae, nrmse, kl_v, mspe
