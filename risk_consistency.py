"""
 
评估模型的风险一致性，用于检测和量化风险失对齐现象

"""


import numpy as np
import json
import os
import argparse
from typing import Tuple


def risk_representation_consistency(embeddings, group_ids):
    """
    Measure intra-group variance of semantic-equivalent samples.
    """
    scores = []

    for gid in np.unique(group_ids):
        group_emb = embeddings[group_ids == gid]
        centroid = group_emb.mean(axis=0)
        variance = np.mean(np.linalg.norm(group_emb - centroid, axis=1))
        scores.append(variance)

    return np.mean(scores)

""" Measure risk score variance within semantic groups. """
def risk_score_stability(risk_scores, group_ids):
    
    variances = []

    for gid in np.unique(group_ids):
        group_scores = risk_scores[group_ids == gid]
        variances.append(np.var(group_scores))

    return np.mean(variances)

"""  综合评估风险一致性 """
def evaluate_risk_consistency(embeddings, risk_scores, group_ids):
    
    # 计算各项指标
    rep_consistency = risk_representation_consistency(embeddings, group_ids)
    score_stability = risk_score_stability(risk_scores, group_ids)
    
    # 计算组间一致性（可选）
    inter_group_variance = calculate_inter_group_variance(embeddings, group_ids)
    
    results = {
        'representation_consistency': float(rep_consistency),
        'risk_score_stability': float(score_stability),
        'inter_group_variance': float(inter_group_variance),
        'num_groups': len(np.unique(group_ids)),
        'total_samples': len(embeddings)
    }
    
    return results

""" 计算组间方差，用于对比组内一致性 """
def calculate_inter_group_variance(embeddings, group_ids):
    
    group_centroids = []
    
    for gid in np.unique(group_ids):
        group_emb = embeddings[group_ids == gid]
        centroid = group_emb.mean(axis=0)
        group_centroids.append(centroid)
    
    group_centroids = np.array(group_centroids)
    overall_centroid = group_centroids.mean(axis=0)
    
    return np.mean(np.linalg.norm(group_centroids - overall_centroid, axis=1))

"""  从文件加载嵌入数据  """
def load_embeddings_from_file(embedding_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if embedding_file.endswith('.json'):
        with open(embedding_file, 'r') as f:
            data = json.load(f)
        
        # 处理隐藏特征JSON格式
        if 'classification_outputs' in data and 'risk_outputs' in data:
            # 从隐藏特征文件提取数据
            risk_outputs = np.array(data['risk_outputs'])
            input_features = np.array(data['input_features'])
            batch_indices = np.array(data['batch_indices'])
            
            # 使用输入特征作为嵌入，风险输出作为风险评分
            # 将输入特征展平为2D数组
            if len(input_features.shape) == 3:  # (batch, seq, features)
                embeddings = input_features.reshape(input_features.shape[0], -1)
            else:
                embeddings = input_features
            
            if len(risk_outputs.shape) == 2:   # 使用风险输出的均值作为风险评分
                risk_scores = np.mean(risk_outputs, axis=1)
            else:
                risk_scores = risk_outputs
            
            group_ids = batch_indices // 4    # 使用批次索引作为组ID（每4个批次为一组）
            
        else:
            # 标准格式
            embeddings = np.array(data['embeddings'])
            risk_scores = np.array(data['risk_scores'])
            group_ids = np.array(data['group_ids'])
    
    elif embedding_file.endswith('.npz'):
        data = np.load(embedding_file)
        embeddings = data['embeddings']
        risk_scores = data['risks']  # 修复键名：'risks' 而不是 'risk_scores'
        group_ids = data['labels']  # 使用labels作为group_ids的替代
    
    else:
        raise ValueError(f"Unsupported file format: {embedding_file}")
    
    return embeddings, risk_scores, group_ids


def main():
    parser = argparse.ArgumentParser(description='Evaluate risk consistency')
    parser.add_argument('--embeddings', type=str, default='E:\\PycharmProjects1\\TWORisk\\nids-risk-misalignment\\analysis\\analysis_results\\cic_ids_test_embeddings.npz', help='Path to embeddings file (.json or .npz)')
    parser.add_argument('--output', type=str, default='E:\\PycharmProjects1\\TWORisk\\nids-risk-misalignment\\analysis\\risk_consistency\\risk_consistency_results.json', help='Output file for results')
    args = parser.parse_args()
    
    # 加载数据
    print(f"Loading embeddings from {args.embeddings}...")
    embeddings, risk_scores, group_ids = load_embeddings_from_file(args.embeddings)
    
    # 评估风险一致性
    print("Evaluating risk consistency...")
    results = evaluate_risk_consistency(embeddings, risk_scores, group_ids)
    
    # 打印结果
    print("\n=== Risk Consistency Results ===")
    print(f"Representation Consistency: {results['representation_consistency']:.6f}")
    print(f"Risk Score Stability: {results['risk_score_stability']:.6f}")
    print(f"Inter-group Variance: {results['inter_group_variance']:.6f}")
    print(f"Number of Groups: {results['num_groups']}")
    print(f"Total Samples: {results['total_samples']}")
    
    # 保存结果
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
