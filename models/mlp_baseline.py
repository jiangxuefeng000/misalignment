"""
MLP 基线模型
提供 MLP 基线模型，用于验证风险解释失对齐现象并非 Transformer 架构特有。
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Dict, Any


class MLPBaseline(keras.Model):
    """MLP 基线模型"""
    
    def __init__(self, config: Dict[str, Any]):
        super(MLPBaseline, self).__init__()
        
        # 模型参数
        self.sequence_length = config.get('sequence_length', 10)
        self.feature_dim = config.get('feature_dim', 64)
        self.hidden_dims = config.get('hidden_dims', [256, 128, 64])
        self.dropout_rate = config.get('dropout_rate', 0.2)
        self.num_classes = config.get('num_classes', 2)
        
        # 输入展平层
        self.flatten = layers.Flatten()
        
        # 共享特征提取器
        self.shared_layers = []
        input_dim = self.sequence_length * self.feature_dim
        
        for hidden_dim in self.hidden_dims:
            self.shared_layers.extend([
                layers.Dense(hidden_dim, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(self.dropout_rate)
            ])
        
        # 共享嵌入层
        self.embedding_dim = self.hidden_dims[-1]
        self.shared_embedding = None
        
        # 攻击分类头
        self.classification_head = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.num_classes, activation='softmax', name='classification')
        ])
        
        # 风险评分头
        self.risk_head = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(1, activation='sigmoid', name='risk_score')
        ])
    
    def call(self, inputs, training=None, return_embedding=False):
        # 输入形状: (batch_size, sequence_length, feature_dim)
        x = self.flatten(inputs)  # (batch_size, sequence_length * feature_dim)
        
        # 通过共享层
        for layer in self.shared_layers:
            x = layer(x, training=training)
        
        # 保存共享嵌入
        self.shared_embedding = x
        
        # 并行输出头
        classification_output = self.classification_head(x, training=training)
        risk_output = self.risk_head(x, training=training)
        
        if return_embedding:
            return classification_output, risk_output, self.shared_embedding
        else:
            return classification_output, risk_output
    
    def get_embedding(self, inputs, training=None):
        """获取中间层嵌入表示"""
        _ = self.call(inputs, training=training, return_embedding=True)
        return self.shared_embedding


class DeepMLPBaseline(MLPBaseline):
    """深度 MLP 基线模型"""
    
    def __init__(self, config: Dict[str, Any]):
        # 增加网络深度
        config['hidden_dims'] = config.get('hidden_dims', [512, 256, 128, 64, 32])
        super(DeepMLPBaseline, self).__init__(config)
        
        # 添加残差连接
        self.residual_layers = []
        for i in range(len(self.hidden_dims) - 1):
            self.residual_layers.append(
                layers.Dense(self.hidden_dims[i + 1], activation='relu')
            )
    
    def call(self, inputs, training=None, return_embedding=False):
        x = self.flatten(inputs)
        
        # 通过共享层，使用残差连接
        for i, layer in enumerate(self.shared_layers):
            if isinstance(layer, layers.Dense):
                x = layer(x, training=training)
                # 添加残差连接（当维度匹配时）
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
    """时序感知 MLP 基线模型"""
    
    def __init__(self, config: Dict[str, Any]):
        super(TemporalMLPBaseline, self).__init__(config)
        
        # 时序特征提取器
        self.temporal_encoder = keras.Sequential([
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.MaxPooling1D(2),
            layers.Conv1D(32, 3, activation='relu', padding='same'),
            layers.GlobalAveragePooling1D()
        ])
        
        # 调整输入维度
        self.feature_dim = config.get('feature_dim', 64)
        self.temporal_dim = 32  # Conv1D 输出维度
        
        # 更新共享层输入维度
        self.shared_layers[0] = layers.Dense(self.hidden_dims[0], activation='relu')
    
    def call(self, inputs, training=None, return_embedding=False):
        # 时序特征提取
        temporal_features = self.temporal_encoder(inputs, training=training)
        
        # 展平原始特征
        flattened = self.flatten(inputs)
        
        # 特征融合
        x = layers.Concatenate()([temporal_features, flattened])
        
        # 通过共享层
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
    """注意力增强 MLP 基线模型"""
    
    def __init__(self, config: Dict[str, Any]):
        super(AttentionMLPBaseline, self).__init__(config)
        
        # 注意力机制
        self.attention_dim = config.get('attention_dim', 64)
        self.sequence_length = config.get('sequence_length', 10)
        
        # 时间步注意力
        self.temporal_attention = layers.Dense(self.attention_dim, activation='tanh')
        self.attention_weights = layers.Dense(1, activation='softmax')
        
        # 特征注意力
        self.feature_attention = layers.MultiHeadAttention(
            num_heads=4, key_dim=self.feature_dim // 4
        )
    
    def call(self, inputs, training=None, return_embedding=False):
        # 时间步注意力
        temporal_scores = self.temporal_attention(inputs)  # (batch, seq, attention_dim)
        attention_weights = self.attention_weights(temporal_scores)  # (batch, seq, 1)
        
        # 加权求和
        weighted_features = tf.reduce_sum(inputs * attention_weights, axis=1)  # (batch, feature_dim)
        
        # 特征注意力
        attention_output = self.feature_attention(
            inputs, inputs, training=training
        )  # (batch, seq, feature_dim)
        
        # 全局池化
        pooled_features = tf.reduce_mean(attention_output, axis=1)  # (batch, feature_dim)
        
        # 特征融合
        x = layers.Concatenate()([weighted_features, pooled_features])
        
        # 展平并处理
        x = self.flatten(x) if len(x.shape) > 2 else x
        
        # 通过共享层
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
    """创建 MLP 基线模型"""
    model_type = config.get('model_type', 'standard')
    
    if model_type == 'deep':
        model = DeepMLPBaseline(config)
    elif model_type == 'temporal':
        model = TemporalMLPBaseline(config)
    elif model_type == 'attention':
        model = AttentionMLPBaseline(config)
    else:
        model = MLPBaseline(config)
    
    # 编译模型
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
    }
    
    # 创建完整模型用于编译
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


def compare_model_complexities(configs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """比较不同模型的复杂度"""
    results = {}
    
    for model_name, config in configs.items():
        model = create_mlp_baseline(config)
        
        # 计算参数数量
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        
        results[model_name] = {
            'total_params': int(total_params),
            'trainable_params': int(trainable_params),
            'non_trainable_params': int(total_params - trainable_params)
        }
    
    return results


if __name__ == "__main__":
    # 示例配置
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
    
    # 比较模型复杂度
    complexity_results = compare_model_complexities(configs)
    
    print("Model Complexity Comparison:")
    for model_name, metrics in complexity_results.items():
        print(f"{model_name}:")
        print(f"  Total params: {metrics['total_params']:,}")
        print(f"  Trainable params: {metrics['trainable_params']:,}")
        print(f"  Non-trainable params: {metrics['non_trainable_params']:,}")
        print()
    
    # 测试模型
    model = create_mlp_baseline(configs['standard_mlp'])
    model.summary()
    
    # 测试预测
    sample_input = np.random.randn(4, 10, 64)
    classification_output, risk_output = model.predict(sample_input)
    
    print(f"Classification output shape: {classification_output.shape}")
    print(f"Risk output shape: {risk_output.shape}")
    
    print("MLP baseline models created successfully!")