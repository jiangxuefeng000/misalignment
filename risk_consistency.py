import numpy as np
import json
import os
import argparse
from typing import Tuple


def risk_representation_consistency(embeddings, group_ids):
    scores = []

    for gid in np.unique(group_ids):
        group_emb = embeddings[group_ids == gid]
        centroid = group_emb.mean(axis=0)
        variance = np.mean(np.linalg.norm(group_emb - centroid, axis=1))
        scores.append(variance)

    return np.mean(scores)

def evaluate_risk_consistency(embeddings, risk_scores, group_ids):

    rep_consistency = risk_representation_consistency(embeddings, group_ids)
    score_stability = risk_score_stability(risk_scores, group_ids)
    inter_group_variance = calculate_inter_group_variance(embeddings, group_ids)
    
    results = {
        'representation_consistency': float(rep_consistency),
        'risk_score_stability': float(score_stability),
        'inter_group_variance': float(inter_group_variance),
        'num_groups': len(np.unique(group_ids)),
        'total_samples': len(embeddings)
    }
    
    return results


def calculate_inter_group_variance(embeddings, group_ids):
    
    group_centroids = []
    
    for gid in np.unique(group_ids):
        group_emb = embeddings[group_ids == gid]
        centroid = group_emb.mean(axis=0)
        group_centroids.append(centroid)
    
    group_centroids = np.array(group_centroids)
    overall_centroid = group_centroids.mean(axis=0)
    
    return np.mean(np.linalg.norm(group_centroids - overall_centroid, axis=1))

def load_embeddings_from_file(embedding_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if embedding_file.endswith('.json'):
        with open(embedding_file, 'r') as f:
            data = json.load(f)
        if 'classification_outputs' in data and 'risk_outputs' in data:
            risk_outputs = np.array(data['risk_outputs'])
            input_features = np.array(data['input_features'])
            batch_indices = np.array(data['batch_indices'])

            if len(input_features.shape) == 3: 
                embeddings = input_features.reshape(input_features.shape[0], -1)
            else:
                embeddings = input_features
            
            if len(risk_outputs.shape) == 2: 
                risk_scores = np.mean(risk_outputs, axis=1)
            else:
                risk_scores = risk_outputs
            
            group_ids = batch_indices // 4 
            
        else:
            embeddings = np.array(data['embeddings'])
            risk_scores = np.array(data['risk_scores'])
            group_ids = np.array(data['group_ids'])
    
    elif embedding_file.endswith('.npz'):
        data = np.load(embedding_file)
        embeddings = data['embeddings']
        risk_scores = data['risks'] 
        group_ids = data['labels'] 
    
    else:
        raise ValueError(f"Unsupported file format: {embedding_file}")
    
    return embeddings, risk_scores, group_ids


def main():
    parser = argparse.ArgumentParser(description='Evaluate risk consistency')
    parser.add_argument('--embeddings', type=str, default='')
    parser.add_argument('--output', type=str, default='')
    args = parser.parse_args()
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()


