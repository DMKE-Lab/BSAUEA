import numpy as np
from scipy.spatial.distance import cosine
from pyemd import emd
from collections import Counter
import networkx as nx

# ---- 参数设置 ----
W_node, W_edge, W_degree, W_scale = 0.25, 0.25, 0.35, 0.15
THRESHOLD = 0.7  # μ

# ---- 辅助函数 ----

def cosine_similarity(v1, v2):
    # 避免0向量问题
    if np.all(v1 == 0) or np.all(v2 == 0):
        return 0.0
    return 1 - cosine(v1, v2)

def normalize_histogram(hist, bins=10):
    total = sum(hist.values())
    vec = np.zeros(bins)
    for k in hist:
        idx = min(k, bins - 1)
        vec[idx] += hist[k]
    return vec / np.sum(vec)

def compute_emd(h1, h2):
    distance_matrix = np.abs(np.subtract.outer(np.arange(len(h1)), np.arange(len(h2)))).astype(np.float64)
    return emd(h1.astype(np.float64), h2.astype(np.float64), distance_matrix)

# ---- 相似性计算函数 ----

def node_type_similarity(G1, G2):
    def get_node_type_vector(G):
        node_types = [data['type'] for _, data in G.nodes(data=True)]
        count = Counter(node_types)
        types = sorted(set(count))
        vec = np.array([count[t] for t in types], dtype=np.float32)
        return vec / vec.sum() if vec.sum() != 0 else vec

    v1 = get_node_type_vector(G1)
    v2 = get_node_type_vector(G2)
    return cosine_similarity(v1, v2)

def edge_type_similarity(G1, G2):
    def get_edge_type_vector(G):
        edge_types = [data['type'] for _, _, data in G.edges(data=True)]
        count = Counter(edge_types)
        types = sorted(set(count))
        vec = np.array([count[t] for t in types], dtype=np.float32)
        return vec / vec.sum() if vec.sum() != 0 else vec

    v1 = get_edge_type_vector(G1)
    v2 = get_edge_type_vector(G2)
    return cosine_similarity(v1, v2)

def degree_distribution_similarity(G1, G2, bins=10):
    def get_degree_hist(G, direction='in'):
        if direction == 'in':
            degrees = [G.in_degree(n) for n in G.nodes()]
        else:
            degrees = [G.out_degree(n) for n in G.nodes()]
        hist = Counter(degrees)
        return normalize_histogram(hist, bins=bins)

    h1_in = get_degree_hist(G1, 'in')
    h1_out = get_degree_hist(G1, 'out')
    h2_in = get_degree_hist(G2, 'in')
    h2_out = get_degree_hist(G2, 'out')

    emd_in = compute_emd(h1_in, h2_in)
    emd_out = compute_emd(h1_out, h2_out)
    return 1 - 0.5 * (emd_in + emd_out)

def graph_scale_similarity(G1, G2):
    n1, e1 = G1.number_of_nodes(), G1.number_of_edges()
    n2, e2 = G2.number_of_nodes(), G2.number_of_edges()
    numerator = abs(n1 - n2) + abs(e1 - e2)
    denominator = n1 + n2 + e1 + e2
    return 1 - (numerator / denominator) if denominator != 0 else 0

# ---- 主函数：结构相似性与GCN激活 ----

def structural_similarity(G_s, G_t):
    sim_node = node_type_similarity(G_s, G_t)
    sim_edge = edge_type_similarity(G_s, G_t)
    sim_degree = degree_distribution_similarity(G_s, G_t)
    sim_scale = graph_scale_similarity(G_s, G_t)

    sim_struct = (W_node * sim_node +
                  W_edge * sim_edge +
                  W_degree * sim_degree +
                  W_scale * sim_scale)

    print(f"Sim_node:   {sim_node:.4f}")
    print(f"Sim_edge:   {sim_edge:.4f}")
    print(f"Sim_degree: {sim_degree:.4f}")
    print(f"Sim_scale:  {sim_scale:.4f}")
    print(f"Sim_struct: {sim_struct:.4f}")

    if sim_struct >= THRESHOLD:
        print("→ Activate GCN for structural aggregation.")
        return True
    else:
        print("→ Skip structural aggregation to avoid noise.")
        return False


def fuse_similarity(struct_sim,
                    embedding_sim_matrix=None,
                    name_sim_matrix=None,
                    temporal_sim_matrix=None,
                    beta=BETA,
                    gamma=GAMMA,
                    threshold=THRESHOLD):
    assert temporal_sim_matrix is not None, "Temporal similarity matrix must be provided."
    assert temporal_sim_matrix.shape[0] == temporal_sim_matrix.shape[1], "Temporal sim matrix must be square."

    if struct_sim >= threshold:
        print(f"结构相似度为 {struct_sim:.2f} ≥ {threshold}, 使用嵌入 + 时间融合")
        assert embedding_sim_matrix is not None, "Embedding similarity matrix required when structure sim >= threshold"
        assert embedding_sim_matrix.shape == temporal_sim_matrix.shape, "Shape mismatch between embedding and temporal sim"
        S_fused = (1 - beta) * embedding_sim_matrix + beta * temporal_sim_matrix
    else:
        print(f"结构相似度为 {struct_sim:.2f} < {threshold}, 使用名称 + 时间融合")
        assert name_sim_matrix is not None, "Name similarity matrix required when structure sim < threshold"
        assert name_sim_matrix.shape == temporal_sim_matrix.shape, "Shape mismatch between name and temporal sim"
        S_fused = (1 - gamma) * name_sim_matrix + gamma * temporal_sim_matrix

    return S_fused