"""
评分模块
独立封装风险评分模块，从架构层面区分攻击判别与风险解释两种功能。
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Dict, Any, Tuple, Optional


class RiskHead(layers.Layer):
    """基础风险评分头"""
    
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
        
        # 构建风险评分网络
        self.risk_layers = []
        
        # 隐藏层
        for i, hidden_dim in enumerate(hidden_dims):
            self.risk_layers.extend([
                layers.Dense(hidden_dim, activation=activation, 
                           name=f'{name}_dense_{i}'),
                layers.BatchNormalization(name=f'{name}_bn_{i}'),
                layers.Dropout(dropout_rate, name=f'{name}_dropout_{i}')
            ])
        
        # 输出层
        self.output_layer = layers.Dense(1, activation='sigmoid', 
                                       name=f'{name}_output')
        
        # 风险解释性层
        self.explanation_layer = layers.Dense(hidden_dims[-1], activation='tanh',
                                           name=f'{name}_explanation')
    
    def call(self, inputs, training=None, return_explanation=False):
        """前向传播"""
        x = inputs
        
        # 通过隐藏层
        for layer in self.risk_layers:
            x = layer(x, training=training)
        
        # 风险评分
        risk_score = self.output_layer(x, training=training)
        
        if return_explanation:
            # 风险解释向量
            explanation = self.explanation_layer(x, training=training)
            return risk_score, explanation
        
        return risk_score
    
    def get_risk_factors(self, inputs, training=None):
        """获取风险因子"""
        risk_score, explanation = self.call(inputs, training=training, return_explanation=True)
        return risk_score, explanation


class CalibratedRiskHead(RiskHead):
    """校准的风险评分头"""
    
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
        
        # 温度校准层
        self.temperature_layer = layers.Lambda(
            lambda x: x / temperature, name=f'{name}_temperature'
        )
    
    def call(self, inputs, training=None, return_explanation=False):
        """前向传播"""
        x = inputs
        
        # 通过隐藏层
        for layer in self.risk_layers:
            x = layer(x, training=training)
        
        # 风险评分（未校准）
        raw_risk = self.output_layer(x, training=training)
        
        # 温度校准
        calibrated_risk = self.temperature_layer(raw_risk)
        
        if return_explanation:
            explanation = self.explanation_layer(x, training=training)
            return calibrated_risk, explanation
        
        return calibrated_risk


class AttentionRiskHead(RiskHead):
    """注意力增强的风险评分头"""
    
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
        
        # 注意力机制
        self.attention_dense = layers.Dense(attention_dim, activation='tanh',
                                         name=f'{name}_attention_dense')
        self.attention_weights = layers.Dense(1, activation='softmax',
                                            name=f'{name}_attention_weights')
        
        # 特征重要性层
        self.feature_importance = layers.Dense(input_dim, activation='sigmoid',
                                            name=f'{name}_feature_importance')
    
    def call(self, inputs, training=None, return_explanation=False):
        """前向传播"""
        # 计算注意力权重
        attention_scores = self.attention_dense(inputs)
        attention_weights = self.attention_weights(attention_scores)
        
        # 加权特征
        weighted_inputs = inputs * attention_weights
        
        # 特征重要性
        feature_importance = self.feature_importance(inputs, training=training)
        
        # 特征选择
        selected_features = weighted_inputs * feature_importance
        
        # 通过标准风险评分层
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
    """集成风险评分头"""
    
    def __init__(self, 
                 input_dim: int,
                 num_experts: int = 3,
                 hidden_dims: list = [128, 64, 32],
                 dropout_rate: float = 0.2,
                 name: str = 'ensemble_risk_head'):
        super(EnsembleRiskHead, self).__init__(name=name)
        
        self.num_experts = num_experts
        self.input_dim = input_dim
        
        # 创建多个专家网络
        self.experts = []
        for i in range(num_experts):
            expert = RiskHead(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate,
                name=f'{name}_expert_{i}'
            )
            self.experts.append(expert)
        
        # 门控网络
        self.gating_network = keras.Sequential([
            layers.Dense(64, activation='relu', name=f'{name}_gate_dense_1'),
            layers.Dropout(dropout_rate, name=f'{name}_gate_dropout'),
            layers.Dense(num_experts, activation='softmax', name=f'{name}_gate_output')
        ])
        
        # 集成输出层
        self.ensemble_output = layers.Dense(1, activation='sigmoid',
                                          name=f'{name}_ensemble_output')
    
    def call(self, inputs, training=None, return_explanation=False):
        """前向传播"""
        # 获取专家预测
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
        
        # 计算门控权重
        gating_weights = self.gating_network(inputs, training=training)
        
        # 加权平均专家预测
        stacked_predictions = tf.stack(expert_predictions, axis=1)  # (batch, num_experts, 1)
        weighted_predictions = tf.reduce_sum(
            stacked_predictions * gating_weights[:, :, tf.newaxis], axis=1
        )
        
        # 最终集成输出
        ensemble_risk = self.ensemble_output(weighted_predictions, training=training)
        
        if return_explanation:
            explanation = {
                'expert_predictions': expert_predictions,
                'gating_weights': gating_weights,
                'expert_explanations': expert_explanations
            }
            return ensemble_risk, explanation
        
        return ensemble_risk


class RiskHeadFactory:
    """风险评分头工厂类"""
    
    @staticmethod
    def create_risk_head(risk_head_type: str, 
                        input_dim: int,
                        config: Dict[str, Any]) -> RiskHead:
        """创建风险评分头"""
        
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
        
        elif risk_head_type == 'ensemble':
            return EnsembleRiskHead(
                input_dim=input_dim,
                num_experts=config.get('num_experts', 3),
                hidden_dims=config.get('hidden_dims', [128, 64, 32]),
                dropout_rate=config.get('dropout_rate', 0.2)
            )
        
        else:
            raise ValueError(f"Unknown risk head type: {risk_head_type}")


def create_risk_head_module(config: Dict[str, Any]) -> RiskHead:
    """创建风险评分模块的便捷函数"""
    risk_head_type = config.get('risk_head_type', 'standard')
    input_dim = config.get('input_dim', 128)
    
    return RiskHeadFactory.create_risk_head(risk_head_type, input_dim, config)


def test_risk_heads():
    """测试不同类型的风险评分头"""
    # 测试配置
    config = {
        'hidden_dims': [64, 32],
        'dropout_rate': 0.2,
        'attention_dim': 32,
        'temperature': 1.5,
        'num_experts': 3
    }
    
    input_dim = 128
    batch_size = 4
    
    # 创建测试输入
    test_input = tf.random.normal((batch_size, input_dim))
    
    # 测试不同类型的风险评分头
    risk_head_types = ['standard', 'calibrated', 'attention', 'ensemble']
    
    print("Testing Risk Heads:")
    print("=" * 50)
    
    for risk_type in risk_head_types:
        print(f"\nTesting {risk_type} risk head:")
        
        try:
            risk_head = RiskHeadFactory.create_risk_head(risk_type, input_dim, config)
            
            # 基本预测
            risk_score = risk_head(test_input)
            print(f"  Risk score shape: {risk_score.shape}")
            print(f"  Risk score range: [{tf.reduce_min(risk_score):.3f}, {tf.reduce_max(risk_score):.3f}]")
            
            # 带解释的预测
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
    
    print("\n" + "=" * 50)
    print("Risk head testing completed!")


if __name__ == "__main__":
    test_risk_heads()