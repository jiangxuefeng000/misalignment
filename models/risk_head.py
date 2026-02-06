
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Dict, Any, Tuple, Optional


class RiskHead(layers.Layer):
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: list = [128, 64, 32],
                 dropout_rate: float = 0.2,
                 activation: str = 'relu',
                 name: str = 'risk_head'):
        super(RiskHead, self).__init__(name=name)
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.risk_layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            self.risk_layers.extend([
                layers.Dense(hidden_dim, activation=activation, 
                           name=f'{name}_dense_{i}'),
                layers.BatchNormalization(name=f'{name}_bn_{i}'),
                layers.Dropout(dropout_rate, name=f'{name}_dropout_{i}')
            ])
        self.output_layer = layers.Dense(1, activation='sigmoid', 
                                       name=f'{name}_output')

        self.explanation_layer = layers.Dense(hidden_dims[-1], activation='tanh',
                                           name=f'{name}_explanation')
    
    def call(self, inputs, training=None, return_explanation=False):
        x = inputs

        for layer in self.risk_layers:
            x = layer(x, training=training)

        risk_score = self.output_layer(x, training=training)
        
        if return_explanation:
            explanation = self.explanation_layer(x, training=training)
            return risk_score, explanation
        
        return risk_score
    
    def get_risk_factors(self, inputs, training=None):
        risk_score, explanation = self.call(inputs, training=training, return_explanation=True)
        return risk_score, explanation


class CalibratedRiskHead(RiskHead):
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: list = [128, 64, 32],
                 dropout_rate: float = 0.2,
                 temperature: float = 1.0,
                 name: str = 'calibrated_risk_head'):
        super(CalibratedRiskHead, self).__init__(
            input_dim, hidden_dims, dropout_rate, name=name
        )
        
        self.temperature = temperature
        
        self.temperature_layer = layers.Lambda(
            lambda x: x / temperature, name=f'{name}_temperature'
        )
    
    def call(self, inputs, training=None, return_explanation=False):
        x = inputs

        for layer in self.risk_layers:
            x = layer(x, training=training)

        raw_risk = self.output_layer(x, training=training)
        calibrated_risk = self.temperature_layer(raw_risk)
        
        if return_explanation:
            explanation = self.explanation_layer(x, training=training)
            return calibrated_risk, explanation
        
        return calibrated_risk


class AttentionRiskHead(RiskHead):
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: list = [128, 64, 32],
                 dropout_rate: float = 0.2,
                 attention_dim: int = 64,
                 name: str = 'attention_risk_head'):
        super(AttentionRiskHead, self).__init__(
            input_dim, hidden_dims, dropout_rate, name=name
        )
        
        self.attention_dim = attention_dim
  
        self.attention_dense = layers.Dense(attention_dim, activation='tanh',
                                         name=f'{name}_attention_dense')
        self.attention_weights = layers.Dense(1, activation='softmax',
                                            name=f'{name}_attention_weights')
       
        self.feature_importance = layers.Dense(input_dim, activation='sigmoid',
                                            name=f'{name}_feature_importance')
    
    def call(self, inputs, training=None, return_explanation=False):
       
        attention_scores = self.attention_dense(inputs)
        attention_weights = self.attention_weights(attention_scores)
        weighted_inputs = inputs * attention_weights
        feature_importance = self.feature_importance(inputs, training=training)
        selected_features = weighted_inputs * feature_importance
        x = selected_features
        for layer in self.risk_layers:
            x = layer(x, training=training)
        
        risk_score = self.output_layer(x, training=training)
        
        if return_explanation:
            explanation = {
                'attention_weights': attention_weights,
                'feature_importance': feature_importance,
                'risk_explanation': self.explanation_layer(x, training=training)
            }
            return risk_score, explanation
        
        return risk_score


class EnsembleRiskHead(layers.Layer):
    
    def __init__(self, 
                 input_dim: int,
                 num_experts: int = 3,
                 hidden_dims: list = [128, 64, 32],
                 dropout_rate: float = 0.2,
                 name: str = 'ensemble_risk_head'):
        super(EnsembleRiskHead, self).__init__(name=name)
        
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.experts = []
        for i in range(num_experts):
            expert = RiskHead(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate,
                name=f'{name}_expert_{i}'
            )
            self.experts.append(expert)
        
        self.ensemble_output = layers.Dense(1, activation='sigmoid',
                                          name=f'{name}_ensemble_output')
    
    def call(self, inputs, training=None, return_explanation=False):
        expert_predictions = []
        expert_explanations = []
        
        for expert in self.experts:
            if return_explanation:
                pred, expl = expert(inputs, training=training, return_explanation=True)
                expert_predictions.append(pred)
                expert_explanations.append(expl)
            else:
                pred = expert(inputs, training=training)
                expert_predictions.append(pred)
        return ensemble_risk


class RiskHeadFactory:
    def create_risk_head(risk_head_type: str, 
                        input_dim: int,
                        config: Dict[str, Any]) -> RiskHead:
        if risk_head_type == 'standard':
            return RiskHead(
                input_dim=input_dim,
                hidden_dims=config.get('hidden_dims', [128, 64, 32]),
                dropout_rate=config.get('dropout_rate', 0.2),
                activation=config.get('activation', 'relu')
            )
        
        elif risk_head_type == 'calibrated':
            return CalibratedRiskHead(
                input_dim=input_dim,
                hidden_dims=config.get('hidden_dims', [128, 64, 32]),
                dropout_rate=config.get('dropout_rate', 0.2),
                temperature=config.get('temperature', 1.0)
            )
        
        elif risk_head_type == 'attention':
            return AttentionRiskHead(
                input_dim=input_dim,
                hidden_dims=config.get('hidden_dims', [128, 64, 32]),
                dropout_rate=config.get('dropout_rate', 0.2),
                attention_dim=config.get('attention_dim', 64)
            )
        
       
        else:
            raise ValueError(f"Unknown risk head type: {risk_head_type}")


def create_risk_head_module(config: Dict[str, Any]) -> RiskHead:
    risk_head_type = config.get('risk_head_type', 'standard')
    input_dim = config.get('input_dim', 128)
    
    return RiskHeadFactory.create_risk_head(risk_head_type, input_dim, config)


def test_risk_heads():
    
    input_dim = 128
    batch_size = 4
    
    test_input = tf.random.normal((batch_size, input_dim))
    risk_head_types = ['standard', 'calibrated', 'attention', 'ensemble']

    for risk_type in risk_head_types:
        try:
            risk_head = RiskHeadFactory.create_risk_head(risk_type, input_dim, config)
            risk_score = risk_head(test_input)
            if hasattr(risk_head, 'call') and 'return_explanation' in risk_head.call.__code__.co_varnames:
                risk_score, explanation = risk_head(test_input, return_explanation=True)
                print(f"  Explanation available: {type(explanation)}")
                
                if isinstance(explanation, dict):
                    for key, value in explanation.items():
                        if hasattr(value, 'shape'):
                            print(f"    {key}: {value.shape}")
                        else:
                            print(f"    {key}: {type(value)}")
            
            print(f"  ✓ {risk_type} risk head works correctly")
            
        except Exception as e:
            print(f"  ✗ {risk_type} risk head failed: {str(e)}")


if __name__ == "__main__":

    test_risk_heads()
