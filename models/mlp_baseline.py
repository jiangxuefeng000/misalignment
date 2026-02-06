

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Dict, Any


class MLPBaseline(keras.Model):
 
    def __init__(self, config: Dict[str, Any]):
        super(MLPBaseline, self).__init__()
        
        self.sequence_length = config.get('sequence_length', 10)
        self.feature_dim = config.get('feature_dim', 64)
        self.hidden_dims = config.get('hidden_dims', [256, 128, 64])
        self.dropout_rate = config.get('dropout_rate', 0.2)
        self.num_classes = config.get('num_classes', 2)

        self.flatten = layers.Flatten()

        self.shared_layers = []
        input_dim = self.sequence_length * self.feature_dim
        
        for hidden_dim in self.hidden_dims:
            self.shared_layers.extend([
                layers.Dense(hidden_dim, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(self.dropout_rate)
            ])
        self.embedding_dim = self.hidden_dims[-1]
        self.shared_embedding = None

        self.classification_head = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.num_classes, activation='softmax', name='classification')
        ])

        self.risk_head = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(1, activation='sigmoid', name='risk_score')
        ])
    
    def call(self, inputs, training=None, return_embedding=False):
        x = self.flatten(inputs)  # (batch_size, sequence_length * feature_dim)

        for layer in self.shared_layers:
            x = layer(x, training=training)
        self.shared_embedding = x

        classification_output = self.classification_head(x, training=training)
        risk_output = self.risk_head(x, training=training)
        
        if return_embedding:
            return classification_output, risk_output, self.shared_embedding
        else:
            return classification_output, risk_output
    
    def get_embedding(self, inputs, training=None):
        _ = self.call(inputs, training=training, return_embedding=True)
        return self.shared_embedding


class DeepMLPBaseline(MLPBaseline):
    
    def __init__(self, config: Dict[str, Any]):
        config['hidden_dims'] = config.get('hidden_dims', [512, 256, 128, 64, 32])
        super(DeepMLPBaseline, self).__init__(config)
        self.residual_layers = []
        for i in range(len(self.hidden_dims) - 1):
            self.residual_layers.append(
                layers.Dense(self.hidden_dims[i + 1], activation='relu')
            )
    
    def call(self, inputs, training=None, return_embedding=False):
        x = self.flatten(inputs)
        for i, layer in enumerate(self.shared_layers):
            if isinstance(layer, layers.Dense):
                x = layer(x, training=training)
                if i < len(self.residual_layers) and i > 0:
                    residual = self.residual_layers[i - 1](x)
                    x = x + 0.1 * residual
            else:
                x = layer(x, training=training)
        
        self.shared_embedding = x
        
        classification_output = self.classification_head(x, training=training)
        risk_output = self.risk_head(x, training=training)
        
        if return_embedding:
            return classification_output, risk_output, self.shared_embedding
        else:
            return classification_output, risk_output


class TemporalMLPBaseline(MLPBaseline):
    
    def __init__(self, config: Dict[str, Any]):
        super(TemporalMLPBaseline, self).__init__(config)
        self.temporal_encoder = keras.Sequential([
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.MaxPooling1D(2),
            layers.Conv1D(32, 3, activation='relu', padding='same'),
            layers.GlobalAveragePooling1D()
        ])

        self.feature_dim = config.get('feature_dim', 64)
        self.temporal_dim = 32  
        self.shared_layers[0] = layers.Dense(self.hidden_dims[0], activation='relu')
    
    def call(self, inputs, training=None, return_embedding=False):
        temporal_features = self.temporal_encoder(inputs, training=training)

        flattened = self.flatten(inputs)
        for layer in self.shared_layers:
            x = layer(x, training=training)
        
        self.shared_embedding = x
        classification_output = self.classification_head(x, training=training)
        risk_output = self.risk_head(x, training=training)
        if return_embedding:
            return classification_output, risk_output, self.shared_embedding
        else:
            return classification_output, risk_output


class AttentionMLPBaseline(MLPBaseline):
    def __init__(self, config: Dict[str, Any]):
        super(AttentionMLPBaseline, self).__init__(config)

        self.attention_dim = config.get('attention_dim', 64)
        self.sequence_length = config.get('sequence_length', 10)
        self.temporal_attention = layers.Dense(self.attention_dim, activation='tanh')
        self.attention_weights = layers.Dense(1, activation='softmax')
        self.feature_attention = layers.MultiHeadAttention(
            num_heads=4, key_dim=self.feature_dim // 4
        )
    
    def call(self, inputs, training=None, return_embedding=False):
        temporal_scores = self.temporal_attention(inputs)  # (batch, seq, attention_dim)
        attention_weights = self.attention_weights(temporal_scores)  # (batch, seq, 1)

        weighted_features = tf.reduce_sum(inputs * attention_weights, axis=1)  # (batch, feature_dim)
        attention_output = self.feature_attention(
            inputs, inputs, training=training
        )
        pooled_features = tf.reduce_mean(attention_output, axis=1)  # (batch, feature_dim)
        for layer in self.shared_layers:
            x = layer(x, training=training)
        
        self.shared_embedding = x
        
        classification_output = self.classification_head(x, training=training)
        risk_output = self.risk_head(x, training=training)
        
        if return_embedding:
            return classification_output, risk_output, self.shared_embedding
        else:
            return classification_output, risk_output


def create_mlp_baseline(config: Dict[str, Any]) -> keras.Model:
    model_type = config.get('model_type', 'standard')
    
    if model_type == 'deep':
        model = DeepMLPBaseline(config)
    elif model_type == 'temporal':
        model = TemporalMLPBaseline(config)
    elif model_type == 'attention':
        model = AttentionMLPBaseline(config)
    else:
        model = MLPBaseline(config)
    optimizer = keras.optimizers.Adam(
        learning_rate=config.get('learning_rate', 0.001)
    )
    
    losses = {
        'classification': keras.losses.SparseCategoricalCrossentropy(),
        'risk_score': keras.losses.BinaryCrossentropy()
    }
    
    loss_weights = {
        'classification': config.get('classification_weight', 1.0),
        'risk_score': config.get('risk_weight', 1.0)
    }
    
    metrics = {
        'classification': ['accuracy'],
        'risk_score': ['mae', 'mse']
    
    sample_input = keras.Input(shape=(config.get('sequence_length', 10), config.get('feature_dim', 64)))
    classification_output, risk_output = model(sample_input)
    
    full_model = keras.Model(
        inputs=sample_input,
        outputs=[classification_output, risk_output],
        name="MLPBaseline_Full"
    )
    
    full_model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics
    )
    
    return full_model



if __name__ == "__main__":
    configs = {
        'standard_mlp': {
            'sequence_length': 10,
            'feature_dim': 64,
            'hidden_dims': [256, 128, 64],
            'dropout_rate': 0.2,
            'num_classes': 2,
            'learning_rate': 0.001,
            'model_type': 'standard'
        },
        'deep_mlp': {
            'sequence_length': 10,
            'feature_dim': 64,
            'hidden_dims': [512, 256, 128, 64, 32],
            'dropout_rate': 0.2,
            'num_classes': 2,
            'learning_rate': 0.001,
            'model_type': 'deep'
        },
        'temporal_mlp': {
            'sequence_length': 10,
            'feature_dim': 64,
            'hidden_dims': [256, 128, 64],
            'dropout_rate': 0.2,
            'num_classes': 2,
            'learning_rate': 0.001,
            'model_type': 'temporal'
        }
    }
    
    complexity_results = compare_model_complexities(configs)
    
    print("Model Complexity Comparison:")
    for model_name, metrics in complexity_results.items():
        print(f"{model_name}:")
        print(f"  Total params: {metrics['total_params']:,}")
        print(f"  Trainable params: {metrics['trainable_params']:,}")
        print(f"  Non-trainable params: {metrics['non_trainable_params']:,}")
        print()

    model = create_mlp_baseline(configs['standard_mlp'])
    model.summary()

    sample_input = np.random.randn(4, 10, 64)
    classification_output, risk_output = model.predict(sample_input)
