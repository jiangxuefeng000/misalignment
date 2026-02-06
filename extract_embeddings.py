import numpy as np
import os
import json
import sys
from typing import Dict, Any
import tensorflow as tf

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.dataset_loader import DatasetLoader
from models.transformer_nids import create_transformer_nids

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

class EmbeddingExtractor: 
    def __init__(self, model, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.batch_size = config.get('batch_size', 32)
  
    def extract_from_dataset_name(self, dataset_name: str) -> Dict[str, Any]:   
        loader = DatasetLoader(self.config)
        datasets = loader.load_datasets(dataset_name)
        
        test_dataset = datasets['test'] 
        
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
            if batch_count >= max_batches:
                print(f"Reached maximum batch limit: {max_batches}")
                break
            
            classification_output, risk_output = self.model(x_batch, training=False)  
            if hasattr(self.model, 'layers'):
                embedding_layer = None
                for layer in reversed(self.model.layers):
                    if 'pool' in layer.name.lower() or 'embedding' in layer.name.lower():
                        embedding_layer = layer
                        break
                    elif 'classifier' in layer.name.lower() and hasattr(layer, 'input'):
                        embedding_layer = layer.input
                        break
                
                if embedding_layer is not None:
                    embedding_model = tf.keras.Model(inputs=self.model.input, outputs=embedding_layer)
                    embeddings = embedding_model(x_batch, training=False)
                else:
                    embeddings = self.model.layers[-2].output 
                    embedding_model = tf.keras.Model(inputs=self.model.input, outputs=embeddings)
                    embeddings = embedding_model(x_batch, training=False)
            else:
                embeddings = self.model.layers[-2].output
                embedding_model = tf.keras.Model(inputs=self.model.input, outputs=embeddings)
                embeddings = embedding_model(x_batch, training=False)

            batch_embeddings = embeddings.numpy()
            batch_risks = risk_output.numpy()
            batch_labels = y_batch.numpy()
            
            all_embeddings.append(batch_embeddings)
            all_risks.append(batch_risks)
            all_labels.append(batch_labels)
            
            batch_count += 1
            total_samples += len(x_batch)

            if batch_count % 100 == 0:
                print(f"Processed {batch_count} batches ({total_samples} samples)...")
        
        print(f"Finished processing {batch_count} batches ({total_samples} samples total)")
        
        return {
            'embeddings': np.vstack(all_embeddings),
            'risks': np.vstack(all_risks),
            'labels': np.concatenate(all_labels)
        }

    def save_embeddings(self, embeddings_data: Dict[str, Any], output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        np.savez(output_path, embeddings=embeddings_data['embeddings'],risks=embeddings_data['risks'], labels=embeddings_data['labels'])  
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

def extract_and_analyze_embeddings(model, datasets: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:

    extractor = EmbeddingExtractor(model, config)
    analyzer = RiskRepresentationAnalyzer(config)
    
    results = {}
    for dataset_name, dataset in datasets.items():
        print(f"\nProcessing {dataset_name} dataset...")
        

        embeddings_data = extractor.extract_from_dataset(dataset, dataset_name)
        embedding_stats = analyzer.analyze_embedding_statistics(embeddings_data)
        risk_stats = analyzer.analyze_risk_distribution(embeddings_data)
        correlation_stats = analyzer.analyze_correlations(embeddings_data)

        results[dataset_name] = {
            'embeddings_data': embeddings_data,
            'embedding_stats': embedding_stats,
            'risk_stats': risk_stats,
            'correlation_stats': correlation_stats
        }
        
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
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
    
    extractor = EmbeddingExtractor(model, config)
    datasets_to_extract = args.datasets  
    embeddings_data = {}
    
    for dataset_name in datasets_to_extract:
        print(f"\nProcessing dataset: {dataset_name}")
        dataset_config = config.copy()
        dataset_config['dataset_name'] = dataset_name
        
        feature_dim = 71  # CIC-IDS 特征维度
        model_path = os.path.dirname(args.model)  

        model_config['feature_dim'] = feature_dim
        model = create_transformer_nids(model_config)
        if os.path.exists(model_path):
            model.load_weights(os.path.join(model_path, 'best_model'))
            print(f"Successfully loaded model weights from {model_path}")
        else:
            print(f"Warning: Model directory {model_path} not found!")
            print(f"Skipping dataset {dataset_name}")
            continue
        extractor = EmbeddingExtractor(model, dataset_config)
        loader = DatasetLoader(dataset_config)
        all_datasets = loader.load_datasets(dataset_name)
        dataset_embeddings = {}
        for split_name in ['train', 'val', 'test']:
            if split_name in all_datasets:
                print(f"Extracting embeddings from {dataset_name}_{split_name} dataset...")
                split_embeddings = extractor.extract_from_dataset(all_datasets[split_name], f"{dataset_name}_{split_name}")
                dataset_embeddings[split_name] = split_embeddings
        for split_name, split_embeddings in dataset_embeddings.items():
            output_path = os.path.join(args.output_dir, f'{dataset_name}_{split_name}_embeddings.npz')
            extractor.save_embeddings(split_embeddings, output_path)
        
        embeddings_data[dataset_name] = dataset_embeddings
    
    print(f"All embeddings extracted and saved to {args.output_dir}")

