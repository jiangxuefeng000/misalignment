"""
表示与语义一致性分析模块
提取模型中间层embedding与评分，作为语义分析的基础数据。
"""

import numpy as np
import os
import json
import sys
from typing import Dict, Any
import tensorflow as tf

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.dataset_loader import DatasetLoader
from models.transformer_nids import create_transformer_nids



""" 提取中间风险表示 """
def extract_embeddings(model, dataset): 
    embeddings = []
    risks = []
    labels = []
    predictions = []

    for x, y in dataset:
        _, risk, emb = model(x, return_embedding=True)
        embeddings.append(emb.numpy())
        risks.append(risk.numpy())
        labels.append(y.numpy())
        predictions.append(tf.argmax(_, axis=-1).numpy())

    return np.vstack(embeddings), np.vstack(risks), np.concatenate(labels)


""" 嵌入表示提取器 """
class EmbeddingExtractor: 
    def __init__(self, model, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.batch_size = config.get('batch_size', 32)
    
    """从数据集名称提取嵌入"""    
    def extract_from_dataset_name(self, dataset_name: str) -> Dict[str, Any]:   
        # 加载数据集
        loader = DatasetLoader(self.config)
        datasets = loader.load_datasets(dataset_name)
        
        test_dataset = datasets['test']   # 使用测试集提取嵌入
        
        return self.extract_from_dataset(test_dataset, f"{dataset_name}_test")
    
    def extract_from_dataset(self, dataset, dataset_name: str = "unknown"):
        print(f"Extracting embeddings from {dataset_name} dataset...")
        
        all_embeddings = []
        all_risks = []
        all_labels = [] 
        batch_count = 0
        total_samples = 0
        max_batches = self.config.get('max_batches', 1000)
        
        for x_batch, y_batch in dataset:
            # 检查批次限制
            if batch_count >= max_batches:
                print(f"Reached maximum batch limit: {max_batches}")
                break
            
            classification_output, risk_output = self.model(x_batch, training=False)  # 获取模型输出 
            
            # 对于Functional模型，需要手动获取嵌入层
            if hasattr(self.model, 'layers'):
                # 查找全局池化层或分类器的前一层
                embedding_layer = None
                for layer in reversed(self.model.layers):
                    if 'pool' in layer.name.lower() or 'embedding' in layer.name.lower():
                        embedding_layer = layer
                        break
                    elif 'classifier' in layer.name.lower() and hasattr(layer, 'input'):
                        embedding_layer = layer.input
                        break
                
                if embedding_layer is not None:
                    # 创建嵌入模型
                    embedding_model = tf.keras.Model(inputs=self.model.input, outputs=embedding_layer)
                    embeddings = embedding_model(x_batch, training=False)
                else:
                    # 备用方案：使用分类器的输入
                    embeddings = self.model.layers[-2].output  # 分类器的前一层
                    embedding_model = tf.keras.Model(inputs=self.model.input, outputs=embeddings)
                    embeddings = embedding_model(x_batch, training=False)
            else:
                # 备用方案：使用分类器的倒数第二层
                embeddings = self.model.layers[-2].output
                embedding_model = tf.keras.Model(inputs=self.model.input, outputs=embeddings)
                embeddings = embedding_model(x_batch, training=False)
            
            # 转换为numpy
            batch_embeddings = embeddings.numpy()
            batch_risks = risk_output.numpy()
            batch_labels = y_batch.numpy()
            
            all_embeddings.append(batch_embeddings)
            all_risks.append(batch_risks)
            all_labels.append(batch_labels)
            
            batch_count += 1
            total_samples += len(x_batch)
            
            # 显示进度
            if batch_count % 100 == 0:
                print(f"Processed {batch_count} batches ({total_samples} samples)...")
        
        print(f"Finished processing {batch_count} batches ({total_samples} samples total)")
        
        return {
            'embeddings': np.vstack(all_embeddings),
            'risks': np.vstack(all_risks),
            'labels': np.concatenate(all_labels)
        }



    """保存嵌入数据"""
    def save_embeddings(self, embeddings_data: Dict[str, Any], output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存为numpy格式
        np.savez(output_path, embeddings=embeddings_data['embeddings'],risks=embeddings_data['risks'], labels=embeddings_data['labels'])  
        # 保存元数据
        metadata = {
            'dataset_name': 'unknown',
            'num_samples': len(embeddings_data['labels']),
            'embedding_dim': embeddings_data['embeddings'].shape[1],
            'config': self.config
        }
        
        metadata_path = output_path.replace('.npz', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Embeddings saved to {output_path}")


"""风险表示分析器"""
class RiskRepresentationAnalyzer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    """分析嵌入统计信息"""
    def analyze_embedding_statistics(self, embeddings_data: Dict[str, Any]) -> Dict[str, Any]:   
        embeddings = embeddings_data['embeddings']
        risks = embeddings_data['risks']
        labels = embeddings_data['labels']
        
        # 分离正常和攻击样本
        normal_mask = labels == 0
        attack_mask = labels == 1
        
        normal_embeddings = embeddings[normal_mask]
        attack_embeddings = embeddings[attack_mask]
        normal_risks = risks[normal_mask]
        attack_risks = risks[attack_mask]
        
        # 计算统计量
        stats = {
            'overall': {
                'mean': np.mean(embeddings, axis=0),
                'std': np.std(embeddings, axis=0),
                'min': np.min(embeddings, axis=0),
                'max': np.max(embeddings, axis=0),
                'norm': np.linalg.norm(embeddings, axis=1).mean()
            },
            'normal': {
                'mean': np.mean(normal_embeddings, axis=0),
                'std': np.std(normal_embeddings, axis=0),
                'min': np.min(normal_embeddings, axis=0),
                'max': np.max(normal_embeddings, axis=0),
                'norm': np.linalg.norm(normal_embeddings, axis=1).mean(),
                'count': len(normal_embeddings),
                'risk_mean': np.mean(normal_risks),
                'risk_std': np.std(normal_risks)
            },
            'attack': {
                'mean': np.mean(attack_embeddings, axis=0),
                'std': np.std(attack_embeddings, axis=0),
                'min': np.min(attack_embeddings, axis=0),
                'max': np.max(attack_embeddings, axis=0),
                'norm': np.linalg.norm(attack_embeddings, axis=1).mean(),
                'count': len(attack_embeddings),
                'risk_mean': np.mean(attack_risks),
                'risk_std': np.std(attack_risks)
            }
        }
        
        # 计算类别间距离
        if len(normal_embeddings) > 0 and len(attack_embeddings) > 0:
            normal_center = stats['normal']['mean']
            attack_center = stats['attack']['mean']
            
            stats['inter_class_distance'] = np.linalg.norm(normal_center - attack_center)
            stats['intra_class_distance_normal'] = self._compute_intra_class_distance(normal_embeddings)
            stats['intra_class_distance_attack'] = self._compute_intra_class_distance(attack_embeddings)
            
            # 计算分离度
            if stats['intra_class_distance_normal'] > 0 and stats['intra_class_distance_attack'] > 0:
                stats['separation_ratio'] = (
                    stats['inter_class_distance'] / 
                    (stats['intra_class_distance_normal'] + stats['intra_class_distance_attack'])
                )
        
        return stats
    
    """计算类内距离"""
    def _compute_intra_class_distance(self, embeddings: np.ndarray) -> float: 
        if len(embeddings) <= 1:
            return 0.0
        
        # 计算所有样本间的平均距离
        center = np.mean(embeddings, axis=0)
        distances = [np.linalg.norm(emb - center) for emb in embeddings]
        
        return np.mean(distances)
    
    
    """分析评分分布"""
    def analyze_risk_distribution(self, embeddings_data: Dict[str, Any]) -> Dict[str, Any]:   
        risks = embeddings_data['risks']
        labels = embeddings_data['labels']
        
        # 分离正常和攻击样本的风险评分
        normal_risks = risks[labels == 0]
        attack_risks = risks[labels == 1]
        
        # 计算分布统计
        risk_stats = {
            'overall': {
                'mean': np.mean(risks),
                'std': np.std(risks),
                'min': np.min(risks),
                'max': np.max(risks),
                'median': np.median(risks),
                'q25': np.percentile(risks, 25),
                'q75': np.percentile(risks, 75)
            },
            'normal': {
                'mean': np.mean(normal_risks) if len(normal_risks) > 0 else 0,
                'std': np.std(normal_risks) if len(normal_risks) > 0 else 0,
                'min': np.min(normal_risks) if len(normal_risks) > 0 else 0,
                'max': np.max(normal_risks) if len(normal_risks) > 0 else 0,
                'median': np.median(normal_risks) if len(normal_risks) > 0 else 0,
                'count': len(normal_risks)
            },
            'attack': {
                'mean': np.mean(attack_risks) if len(attack_risks) > 0 else 0,
                'std': np.std(attack_risks) if len(attack_risks) > 0 else 0,
                'min': np.min(attack_risks) if len(attack_risks) > 0 else 0,
                'max': np.max(attack_risks) if len(attack_risks) > 0 else 0,
                'median': np.median(attack_risks) if len(attack_risks) > 0 else 0,
                'count': len(attack_risks)
            }
        }
        
        # 计算区分度
        if len(normal_risks) > 0 and len(attack_risks) > 0:
            risk_stats['risk_separation'] = (
                np.mean(attack_risks) - np.mean(normal_risks)
            ) / np.sqrt(np.var(normal_risks) + np.var(attack_risks))
        
        return risk_stats
    
    
    """分析嵌入维度与风险评分的相关性"""
    def analyze_correlations(self, embeddings_data: Dict[str, Any]) -> Dict[str, Any]:
        embeddings = embeddings_data['embeddings']
        risks = embeddings_data['risks']
        
        # 计算每个嵌入维度与评分的相关性
        correlations = []
        for dim in range(embeddings.shape[1]):
            dim_values = embeddings[:, dim]
            correlation = np.corrcoef(dim_values, risks)[0, 1]
            correlations.append(correlation)
        
        correlations = np.array(correlations)
        
        # 找出最相关的维度
        top_corr_indices = np.argsort(np.abs(correlations))[-10:][::-1]
        
        correlation_analysis = {
            'mean_correlation': np.mean(np.abs(correlations)),
            'max_correlation': np.max(np.abs(correlations)),
            'min_correlation': np.min(np.abs(correlations)),
            'std_correlation': np.std(np.abs(correlations)),
            'top_dimensions': [
                {
                    'dimension': int(idx),
                    'correlation': float(correlations[idx]),
                    'abs_correlation': float(abs(correlations[idx]))
                }
                for idx in top_corr_indices
            ],
            'all_correlations': correlations.tolist()
        }
        
        return correlation_analysis

"""提取并分析嵌入的便捷函数"""
def extract_and_analyze_embeddings(model, datasets: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    
    # 创建提取器
    extractor = EmbeddingExtractor(model, config)
    analyzer = RiskRepresentationAnalyzer(config)
    
    results = {}
    
    # 对每个数据集进行提取和分析
    for dataset_name, dataset in datasets.items():
        print(f"\nProcessing {dataset_name} dataset...")
        
        # 提取嵌入
        embeddings_data = extractor.extract_from_dataset(dataset, dataset_name)
        
        # 分析嵌入
        embedding_stats = analyzer.analyze_embedding_statistics(embeddings_data)
        risk_stats = analyzer.analyze_risk_distribution(embeddings_data)
        correlation_stats = analyzer.analyze_correlations(embeddings_data)
        
        # 保存结果
        results[dataset_name] = {
            'embeddings_data': embeddings_data,
            'embedding_stats': embedding_stats,
            'risk_stats': risk_stats,
            'correlation_stats': correlation_stats
        }
        
        # 保存嵌入数据
        output_path = os.path.join(
            config.get('output_dir', 'analysis_results'),
            f'{dataset_name}_embeddings.npz'
        )
        extractor.save_embeddings(embeddings_data, output_path)
    
    return results


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Extract embeddings from trained models')
    parser.add_argument('--model', type=str, default='experiments/risk_weighted_20260121_183517/best_model',
                   help='Path to trained model weights')
    parser.add_argument('--datasets', type=str, nargs='+', 
                       default=['cic_ids'], help='Dataset names to process (default: cic_ids)')
    parser.add_argument('--output-dir', type=str, default='E:\\PycharmProjects1\\TWORisk\\nids-risk-misalignment\\analysis\\analysis_results',
                       help='Output directory for embeddings')
    parser.add_argument('--batch-size', type=int, default=16, 
                   help='Batch size for extraction')
    parser.add_argument('--max-batches', type=int, default=10000,
                       help='Maximum number of batches to process (default: 1000)')
    parser.add_argument('--use-gpu', action='store_true', default=True,
                       help='Use GPU for extraction (default: True)')
    
    args = parser.parse_args()
    
    # 确保在正确的项目目录中
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    # 添加项目根目录到Python路径
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # 配置GPU使用 - 简化版本
    if args.use_gpu:
        print("Using GPU for embedding extraction")
    else:
        print("Using CPU for embedding extraction")

    config = {
        'processed_data_dir': os.path.join(project_root, 'data', 'processed_data'),
        'semantic_data_dir': os.path.join(project_root, 'data', 'semantic_sets_data'),
        'batch_size': args.batch_size,
        'output_dir': args.output_dir,
        'max_batches': args.max_batches
    }
    
    print(f"Loading model from {args.model}...")
    print(f"Project root: {project_root}")
    print(f"Using device: {'GPU' if args.use_gpu else 'CPU'}")
    
    # 尝试从模型目录加载配置
    model_dir = os.path.dirname(args.model)
    config_file = os.path.join(model_dir, 'config.json')
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            model_config = json.load(f)
        print(f"Loaded model config from {config_file}")
    else:
        model_config = {
            'sequence_length': 10,
            'feature_dim': 71,  
            'd_model': 64,
            'num_heads': 2,
            'num_layers': 2,
            'num_classes': 2
        }
        print(f"Using default model config")
    
    print(f"Model config: {model_config}")
    
    
    
    model = create_transformer_nids(model_config)
    model.load_weights(args.model)
    
    print("Model loaded successfully!")
    
    # 提取嵌入
    extractor = EmbeddingExtractor(model, config)
    
    # 提取所有数据集
    datasets_to_extract = args.datasets  # 使用用户指定的数据集列表
    embeddings_data = {}
    
    for dataset_name in datasets_to_extract:
        print(f"\nProcessing dataset: {dataset_name}")
        
        # 为每个数据集创建配置
        dataset_config = config.copy()
        dataset_config['dataset_name'] = dataset_name
        
        # 设置特征维度和模型路径（只支持cic_ids）
        feature_dim = 71  # CIC-IDS 特征维度
        model_path = os.path.dirname(args.model)  # 使用默认模型目录
        
        # 更新模型配置
        model_config['feature_dim'] = feature_dim
        print(f"Using feature_dim={feature_dim} for dataset {dataset_name}")
        print(f"Loading model weights from {model_path}")
        
        # 重新创建模型以适应新的特征维度
        model = create_transformer_nids(model_config)
        
        # 检查模型文件是否存在（检查目录）
        if os.path.exists(model_path):
            model.load_weights(os.path.join(model_path, 'best_model'))
            print(f"Successfully loaded model weights from {model_path}")
        else:
            print(f"Warning: Model directory {model_path} not found!")
            print(f"Skipping dataset {dataset_name}")
            continue
        
        # 创建新的提取器
        extractor = EmbeddingExtractor(model, dataset_config)
        
        # 加载当前数据集
        loader = DatasetLoader(dataset_config)
        all_datasets = loader.load_datasets(dataset_name)
        
        # 提取当前数据集的所有分割
        dataset_embeddings = {}
        for split_name in ['train', 'val', 'test']:
            if split_name in all_datasets:
                print(f"Extracting embeddings from {dataset_name}_{split_name} dataset...")
                split_embeddings = extractor.extract_from_dataset(all_datasets[split_name], f"{dataset_name}_{split_name}")
                dataset_embeddings[split_name] = split_embeddings
        
        # 保存当前数据集的所有分割
        for split_name, split_embeddings in dataset_embeddings.items():
            output_path = os.path.join(args.output_dir, f'{dataset_name}_{split_name}_embeddings.npz')
            extractor.save_embeddings(split_embeddings, output_path)
        
        embeddings_data[dataset_name] = dataset_embeddings
    
    print(f"All embeddings extracted and saved to {args.output_dir}")
