import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import  Dict, Any


class MultiHeadAttention(layers.Layer):
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
        attn_output = self.mha(inputs, training=training, mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


class PositionalEncoding(layers.Layer):
    
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
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        
        return tf.cast(pos_encoding, tf.float32)
    
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


class TransformerNIDS(keras.Model):
    def __init__(self, feature_dim, d_model=128, num_heads=4, num_layers=3, num_classes=2, **kwargs):
        super(TransformerNIDS, self).__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.input_proj = layers.Dense(d_model)
        self.pos_encoding = PositionalEncoding(1000, d_model)  
        self.transformer_blocks = [
            TransformerEncoder(d_model, num_heads, d_model * 4, 0.1)
            for _ in range(num_layers)
        ]

        self.dropout = layers.Dropout(0.1)
        self.global_pool = layers.GlobalAveragePooling1D()
        self.classifier = layers.Dense(num_classes, activation="softmax")
        self.risk_head = layers.Dense(1, activation="sigmoid")
        self.shared_embedding = None

    def call(self, x, training=False, return_embedding=False):
        
        h = self.input_proj(x)
        h = self.pos_encoding(h)
        for block in self.transformer_blocks:
            h = block(h, training=training)
        
        h = self.dropout(h, training=training)
        embedding = self.global_pool(h)
        self.shared_embedding = embedding
        cls_out = self.classifier(embedding)
        risk_score = self.risk_head(embedding)

        if return_embedding:
            return cls_out, risk_score, embedding

        return cls_out, risk_score
    
    def get_embedding(self, inputs, training=None):
        _ = self.call(inputs, training=training, return_embedding=True)
        return self.shared_embedding

def create_transformer_nids(config: Dict[str, Any]) -> keras.Model:
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
    import json
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return create_transformer_nids(config)


if __name__ == "__main__":
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
    
    model = create_transformer_nids(config)

    sample_input = np.random.randn(4, 10, 64)
    classification_output, risk_output = model.predict(sample_input)

