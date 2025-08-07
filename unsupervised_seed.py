import numpy as np
from scipy.special import softmax

# ---------- 参数设置 ----------
DELTA = 0.5  # 时间相似度权重 δ
EPSILON = 0.5  # 高置信度实体对的 ε 阈值
OVERLAP_THRESHOLD = 0.75  # 75% 判定是否用 Sinkhorn
TEMPERATURE = 0.1  # Sinkhorn softmax温度系数
SINKHORN_ITER = 20  # Sinkhorn迭代次数


# ---------- 时间相似度 + GloVe相似度聚合 ----------
def compute_aggregated_similarity(glove_sim, temporal_sim, delta=DELTA):
    return (1 - delta) * glove_sim + delta * temporal_sim


# ---------- 获取Top-1匹配实体对 ----------
def get_confident_pairs(sim_matrix, threshold=EPSILON):
    confident_pairs = []
    for i in range(sim_matrix.shape[0]):
        top_idx = np.argmax(sim_matrix[i])
        second_val = np.partition(sim_matrix[i], -2)[-2]
        top_val = sim_matrix[i][top_idx]
        reverse_top_idx = np.argmax(sim_matrix[:, top_idx])
        # 双向Top-1 且 ε 差值足够大
        if reverse_top_idx == i and (top_val - second_val) > threshold:
            confident_pairs.append((i, top_idx))
    return confident_pairs


# ---------- Sinkhorn算法 ----------
def sinkhorn(S, T=TEMPERATURE, q=SINKHORN_ITER):
    S_scaled = np.exp(S / T)
    for _ in range(q):
        S_scaled = S_scaled / S_scaled.sum(axis=1, keepdims=True)  # row norm
        S_scaled = S_scaled / S_scaled.sum(axis=0, keepdims=True)  # col norm
    return S_scaled


# ---------- Sinkhorn取Top-1近似匹配 ----------
def get_sinkhorn_pairs(sim_matrix):
    P = sinkhorn(sim_matrix)
    return list(zip(*np.where(P == P.max(axis=1, keepdims=True))))  # Top-1 row match


# ---------- 主函数 ----------
def generate_alignment_seeds(glove_sim, temporal_sim, overlap_ratio):
    agg_sim = compute_aggregated_similarity(glove_sim, temporal_sim)
    if overlap_ratio > OVERLAP_THRESHOLD:
        print("→ 使用 Sinkhorn 算法进行近似一一匹配")
        matched_pairs = get_sinkhorn_pairs(agg_sim)
    else:
        print("→ 使用 Top-1 + 置信度差值筛选匹配")
        matched_pairs = get_confident_pairs(agg_sim)

    print(f"生成伪对齐种子数：{len(matched_pairs)}")
    return matched_pairs



