"""
基于 Transformer 的网络入侵检测系统
实现基于 Transformer 的 NIDS，包含共享编码器与两个并行输出头：攻击分类头与评分头。
中间层 embedding 被显式保留，作为语义表示。
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import  Dict, Any


class MultiHeadAttention(layers.Layer):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        
        assert d_model % num_heads == 0
        
        self.query_dense = layers.Dense(d_model)
        self.key_dense = layers.Dense(d_model)
        self.value_dense = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)
        
        self.dropout = layers.Dropout(dropout_rate)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs, training=None, mask=None):
        batch_size = tf.shape(inputs)[0]
        
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        
        # 缩放点积注意力
        scaled_attention = tf.matmul(query, key, transpose_b=True)
        scaled_attention = scaled_attention / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        
        if mask is not None:
            scaled_attention += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaled_attention, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)
        
        attention_output = tf.matmul(attention_weights, value)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, -1, self.d_model))
        
        output = self.dense(attention_output)
        
        return output


class TransformerEncoder(layers.Layer):
    """Transformer 编码器层"""
    
    def __init__(self, d_model: int, num_heads: int, dff: int, dropout_rate: float = 0.1):
        super(TransformerEncoder, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.ffn = self.point_wise_feed_forward_network(d_model, dff)
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def point_wise_feed_forward_network(self, d_model, dff):
        return keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])
    
    def call(self, inputs, training=None, mask=None):
        # 多头注意力
        attn_output = self.mha(inputs, training=training, mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # 前馈网络
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


class PositionalEncoding(layers.Layer):
    """位置编码"""
    
    def __init__(self, position: int, d_model: int):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)
    
    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles
    
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )
        
        # 对偶数位置应用sin
        sines = tf.math.sin(angle_rads[:, 0::2])
        # 对奇数位置应用cos
        cosines = tf.math.cos(angle_rads[:, 1::2])
        
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        
        return tf.cast(pos_encoding, tf.float32)
    
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


class TransformerNIDS(keras.Model):
    """基于 Transformer 的网络入侵检测系统"""
    
    def __init__(self, feature_dim, d_model=128, num_heads=4, num_layers=3, num_classes=2, **kwargs):
        super(TransformerNIDS, self).__init__()
        
        # 模型参数
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # 输入投影层
        self.input_proj = layers.Dense(d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(1000, d_model)  # 假设最大序列长度为1000
        
        # Transformer 编码器层
        self.transformer_blocks = [
            TransformerEncoder(d_model, num_heads, d_model * 4, 0.1)
            for _ in range(num_layers)
        ]
        
        # Dropout层
        self.dropout = layers.Dropout(0.1)
        
        # 全局池化
        self.global_pool = layers.GlobalAveragePooling1D()
        
        # 攻击分类头
        self.classifier = layers.Dense(num_classes, activation="softmax")
        
        # 风险评分头
        self.risk_head = layers.Dense(1, activation="sigmoid")
        
        # 共享嵌入
        self.shared_embedding = None

    def call(self, x, training=False, return_embedding=False):
        """
        Args:
            x: [B, T, F] flow feature sequence
        """
        # 输入投影
        h = self.input_proj(x)
        
        # 位置编码
        h = self.pos_encoding(h)
        
        # Transformer编码器
        for block in self.transformer_blocks:
            h = block(h, training=training)
        
        h = self.dropout(h, training=training)
        
        # 全局池化获得嵌入
        embedding = self.global_pool(h)
        self.shared_embedding = embedding
        
        # 分类和风险评分
        cls_out = self.classifier(embedding)
        risk_score = self.risk_head(embedding)

        if return_embedding:
            return cls_out, risk_score, embedding

        return cls_out, risk_score
    
    def get_embedding(self, inputs, training=None):
        """获取中间层嵌入表示"""
        _ = self.call(inputs, training=training, return_embedding=True)
        return self.shared_embedding


class RiskAwareTransformerNIDS(TransformerNIDS):
    """风险感知的 Transformer NIDS"""
    
    def __init__(self, feature_dim, d_model=128, num_heads=4, num_layers=3, num_classes=2, **kwargs):
        super(RiskAwareTransformerNIDS, self).__init__(
            feature_dim, d_model, num_heads, num_layers, num_classes
        )
        
        # 风险感知注意力机制
        self.risk_attention = MultiHeadAttention(d_model, num_heads // 2, 0.1)
        
        # 风险特征融合层
        self.risk_fusion = layers.Dense(d_model, activation='tanh')
        
    def call(self, x, training=False, return_embedding=False):
        # 基础Transformer处理
        h = self.input_proj(x)
        h = self.pos_encoding(h)
        
        for block in self.transformer_blocks:
            h = block(h, training=training)
        
        # 风险感知注意力
        risk_context = self.risk_attention(h, training=training)
        risk_context = self.risk_fusion(risk_context)
        
        # 融合原始特征和风险上下文
        h = h + 0.1 * risk_context  # 可学习的融合权重
        h = self.dropout(h, training=training)
        
        # 全局池化获得嵌入
        embedding = self.global_pool(h)
        self.shared_embedding = embedding
        
        # 分类和风险评分
        cls_out = self.classifier(embedding)
        risk_score = self.risk_head(embedding)

        if return_embedding:
            return cls_out, risk_score, embedding

        return cls_out, risk_score


def create_transformer_nids(config: Dict[str, Any]) -> keras.Model:
    """创建 Transformer NIDS 模型"""
    model_type = config.get('model_type', 'standard')
    
    if model_type == 'risk_aware':
        model = RiskAwareTransformerNIDS(
            feature_dim=config.get('feature_dim', 64),
            d_model=config.get('d_model', 128),
            num_heads=config.get('num_heads', 4),
            num_layers=config.get('num_layers', 3),
            num_classes=config.get('num_classes', 2)
        )
    else:
        model = TransformerNIDS(
            feature_dim=config.get('feature_dim', 64),
            d_model=config.get('d_model', 128),
            num_heads=config.get('num_heads', 4),
            num_layers=config.get('num_layers', 3),
            num_classes=config.get('num_classes', 2)
        )
    
    # 编译模型
    optimizer = keras.optimizers.Adam(
        learning_rate=config.get('learning_rate', 0.001),
        clipnorm=config.get('clip_norm', 1.0)
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
        name="TransformerNIDS_Full"
    )
    
    full_model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics
    )
    
    return full_model


def create_transformer_nids_from_config(config_path: str) -> keras.Model:
    """从配置文件创建模型"""
    import json
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return create_transformer_nids(config)


if __name__ == "__main__":
    # 示例配置
    config = {
        'sequence_length': 10,
        'feature_dim': 64,
        'd_model': 128,
        'num_heads': 4,
        'num_layers': 3,
        'num_classes': 2,
        'learning_rate': 0.001,
        'model_type': 'standard'
    }
    
    # 创建模型
    model = create_transformer_nids(config)
    
    # 打印模型摘要
    model.summary()
    
    # 测试模型
    sample_input = np.random.randn(4, 10, 64)
    classification_output, risk_output = model.predict(sample_input)
    
    print(f"Classification output shape: {classification_output.shape}")
    print(f"Risk output shape: {risk_output.shape}")
    
    print("Transformer NIDS model created successfully!")
