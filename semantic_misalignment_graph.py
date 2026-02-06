"""
语义错位传播图 (Semantic Misalignment Propagation Graph)
用于追踪Transformer模型在入侵检测中语义错位如何通过层间传播最终导致分类错误

核心功能：
1. 计算每层的语义错位度量
2. 追踪正确/错误分类样本的特征流
3. 生成热力图背景显示错位程度
4. 分面展示误报和漏报样本
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, RegularPolygon
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
import os
import warnings
warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False


class SemanticMisalignmentAnalyzer:
    """语义错位分析器"""
    
    def __init__(self, layer_features, labels, predictions, layer_names=None):
        """
        初始化分析器
        
        Args:
            layer_features: 各层特征列表 [(layer_name, features), ...]
            labels: 真实标签 (0=Normal, 1=Attack)
            predictions: 模型预测标签
            layer_names: 层名称列表（可选）
        """
        self.layer_features = layer_features
        self.labels = np.array(labels)
        self.predictions = np.array(predictions)
        self.layer_names = layer_names or [name for name, _ in layer_features]
        self.num_layers = len(layer_features)
        
        # 分类结果
        self.correct_mask = self.labels == self.predictions
        self.incorrect_mask = ~self.correct_mask
        
        # 误报和漏报
        self.false_positive_mask = (self.labels == 0) & (self.predictions == 1)  # Normal被误判为Attack
        self.false_negative_mask = (self.labels == 1) & (self.predictions == 0)  # Attack被漏报为Normal
        
        # 计算各层错位度量
        self.layer_misalignment = self._compute_layer_misalignment()
        self.layer_class_distances = self._compute_class_distances()
        self.layer_intra_variances = self._compute_intra_class_variance()
        
    def _preprocess_features(self, features):
        """预处理特征（处理3D特征）"""
        if len(features.shape) == 3:
            return features.mean(axis=1)  # 对序列维度取平均
        return features
    
    def _compute_layer_misalignment(self):
        """计算每层的语义错位度量"""
        misalignment_scores = []
        
        for layer_name, features in self.layer_features:
            features = self._preprocess_features(features)
            
            # 计算错误分类样本与正确分类样本的特征差异
            if np.sum(self.incorrect_mask) > 0 and np.sum(self.correct_mask) > 0:
                correct_features = features[self.correct_mask]
                incorrect_features = features[self.incorrect_mask]
                
                # 错位度量：错误样本到正确样本中心的平均距离
                correct_center = correct_features.mean(axis=0)
                incorrect_distances = np.linalg.norm(incorrect_features - correct_center, axis=1)
                misalignment = incorrect_distances.mean()
                
                # 归一化
                all_distances = np.linalg.norm(features - correct_center, axis=1)
                misalignment_normalized = misalignment / (all_distances.mean() + 1e-8)
            else:
                misalignment_normalized = 0.0
            
            misalignment_scores.append(misalignment_normalized)
        
        return np.array(misalignment_scores)
    
    def _compute_class_distances(self):
        """计算每层的类间距离"""
        distances = []
        
        for layer_name, features in self.layer_features:
            features = self._preprocess_features(features)
            
            normal_features = features[self.labels == 0]
            attack_features = features[self.labels == 1]
            
            if len(normal_features) > 0 and len(attack_features) > 0:
                normal_center = normal_features.mean(axis=0)
                attack_center = attack_features.mean(axis=0)
                distance = np.linalg.norm(normal_center - attack_center)
            else:
                distance = 0.0
            
            distances.append(distance)
        
        return np.array(distances)
    
    def _compute_intra_class_variance(self):
        """计算每层的类内方差"""
        variances = []
        
        for layer_name, features in self.layer_features:
            features = self._preprocess_features(features)
            
            normal_features = features[self.labels == 0]
            attack_features = features[self.labels == 1]
            
            normal_var = np.var(normal_features, axis=0).mean() if len(normal_features) > 0 else 0
            attack_var = np.var(attack_features, axis=0).mean() if len(attack_features) > 0 else 0
            
            variances.append((normal_var + attack_var) / 2)
        
        return np.array(variances)
    
    def _compute_sample_trajectories(self, sample_indices, n_components=2):
        """计算样本在各层的轨迹（降维后）"""
        trajectories = []
        
        for idx in sample_indices:
            trajectory = []
            for layer_name, features in self.layer_features:
                features = self._preprocess_features(features)
                trajectory.append(features[idx])
            trajectories.append(trajectory)
        
        return trajectories
    
    def plot_misalignment_propagation_graph(self, save_dir, max_samples=100):
        """
        绘制语义错位传播图
        
        Args:
            save_dir: 保存目录
            max_samples: 最大样本数（用于可视化）
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建大图
        fig = plt.figure(figsize=(24, 16))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1],
                              hspace=0.3, wspace=0.25)
        
        # ========== 主图：语义错位传播热力图 ==========
        ax_main = fig.add_subplot(gs[0, :])
        self._plot_main_propagation_graph(ax_main, max_samples)
        
        # ========== 左下：误报样本流（Normal→Attack误判） ==========
        ax_fp = fig.add_subplot(gs[1, 0])
        self._plot_sample_flow(ax_fp, self.false_positive_mask, 
                              "False Positive Flow (Normal → Attack Misclassified)",
                              max_samples=max_samples//2)
        
        # ========== 右下：漏报样本流（Attack→Normal漏报） ==========
        ax_fn = fig.add_subplot(gs[1, 1])
        self._plot_sample_flow(ax_fn, self.false_negative_mask,
                              "False Negative Flow (Attack → Normal Missed)",
                              max_samples=max_samples//2)
        
        plt.suptitle('Semantic Misalignment Propagation Graph\nTransformer Intrusion Detection Analysis',
                    fontsize=20, fontweight='bold', y=0.98)
        
        save_path = os.path.join(save_dir, 'semantic_misalignment_propagation.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"语义错位传播图已保存到: {save_path}")
        
        # 绘制详细分析图
        self._plot_detailed_analysis(save_dir)
        
        return save_path
    
    def _plot_main_propagation_graph(self, ax, max_samples):
        """绘制主传播图"""
        num_layers = self.num_layers
        
        # 创建热力图背景（错位程度）
        misalignment_normalized = (self.layer_misalignment - self.layer_misalignment.min()) / \
                                  (self.layer_misalignment.max() - self.layer_misalignment.min() + 1e-8)
        
        # 绘制层级背景
        layer_width = 1.0
        layer_height = 0.8
        layer_spacing = 1.5
        
        # 颜色映射：绿色（低错位）→ 黄色 → 红色（高错位）
        cmap = LinearSegmentedColormap.from_list('misalignment', 
                                                  ['#2ecc71', '#f1c40f', '#e74c3c'])
        
        # 绘制每层的背景和标签
        for i in range(num_layers):
            x = i * layer_spacing
            color = cmap(misalignment_normalized[i])
            
            # 层背景矩形
            rect = FancyBboxPatch((x - layer_width/2, -layer_height/2), 
                                  layer_width, layer_height,
                                  boxstyle="round,pad=0.05,rounding_size=0.1",
                                  facecolor=color, edgecolor='black', linewidth=2,
                                  alpha=0.8)
            ax.add_patch(rect)
            
            # 层名称
            short_name = self.layer_names[i].replace('Transformer ', 'T').replace('Layer ', 'L')
            ax.text(x, layer_height/2 + 0.15, short_name, ha='center', va='bottom',
                   fontsize=11, fontweight='bold')
            
            # 错位度量值
            ax.text(x, 0, f'{self.layer_misalignment[i]:.3f}', ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white')
            
            # 标记关键层
            if i > 0 and misalignment_normalized[i] - misalignment_normalized[i-1] > 0.2:
                # 错位突增层 - 黄色警告
                warning = RegularPolygon((x, -layer_height/2 - 0.2), numVertices=3, 
                                        radius=0.12, orientation=0,
                                        facecolor='#f39c12', edgecolor='black')
                ax.add_patch(warning)
                ax.text(x, -layer_height/2 - 0.35, '!', ha='center', va='center',
                       fontsize=10, fontweight='bold', color='black')
            
            if misalignment_normalized[i] == 1.0:
                # 最高错位层 - 红色爆炸图标
                explosion = RegularPolygon((x, layer_height/2 + 0.35), numVertices=8,
                                          radius=0.15, orientation=np.pi/8,
                                          facecolor='#e74c3c', edgecolor='darkred')
                ax.add_patch(explosion)
        
        # 绘制样本流箭头
        correct_count = np.sum(self.correct_mask)
        incorrect_count = np.sum(self.incorrect_mask)
        total_count = len(self.labels)
        
        # 正确分类流（绿色实线）
        for i in range(num_layers - 1):
            x1, x2 = i * layer_spacing + layer_width/2, (i+1) * layer_spacing - layer_width/2
            arrow_width = max(0.5, 3 * correct_count / total_count)
            ax.annotate('', xy=(x2, 0.15), xytext=(x1, 0.15),
                       arrowprops=dict(arrowstyle='->', color='#27ae60', 
                                      lw=arrow_width, mutation_scale=15))
        
        # 错误分类流（红色虚线）
        for i in range(num_layers - 1):
            x1, x2 = i * layer_spacing + layer_width/2, (i+1) * layer_spacing - layer_width/2
            arrow_width = max(0.5, 3 * incorrect_count / total_count)
            ax.annotate('', xy=(x2, -0.15), xytext=(x1, -0.15),
                       arrowprops=dict(arrowstyle='->', color='#e74c3c',
                                      lw=arrow_width, linestyle='--', mutation_scale=15))
        
        # 最终分类结果标记
        final_x = (num_layers - 1) * layer_spacing + layer_width/2 + 0.3
        
        # 正确分类标记（绿色勾）
        ax.text(final_x, 0.3, '✓', fontsize=24, color='#27ae60', fontweight='bold',
               ha='left', va='center')
        ax.text(final_x + 0.3, 0.3, f'Correct: {correct_count}', fontsize=11,
               ha='left', va='center', color='#27ae60')
        
        # 错误分类标记（红色X）
        ax.text(final_x, -0.3, '✗', fontsize=24, color='#e74c3c', fontweight='bold',
               ha='left', va='center')
        ax.text(final_x + 0.3, -0.3, f'Incorrect: {incorrect_count}', fontsize=11,
               ha='left', va='center', color='#e74c3c')
        
        # 添加图例
        legend_elements = [
            mpatches.Patch(facecolor='#27ae60', edgecolor='black', label='Low Misalignment'),
            mpatches.Patch(facecolor='#f1c40f', edgecolor='black', label='Medium Misalignment'),
            mpatches.Patch(facecolor='#e74c3c', edgecolor='black', label='High Misalignment'),
            plt.Line2D([0], [0], color='#27ae60', lw=3, label='Correct Classification Flow'),
            plt.Line2D([0], [0], color='#e74c3c', lw=3, linestyle='--', label='Incorrect Classification Flow'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)
        
        # 设置坐标轴
        ax.set_xlim(-1, num_layers * layer_spacing + 1)
        ax.set_ylim(-1, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Layer-wise Semantic Misalignment Propagation', fontsize=14, fontweight='bold', pad=10)
    
    def _plot_sample_flow(self, ax, mask, title, max_samples=50):
        """绘制特定类型样本的流动图"""
        indices = np.where(mask)[0]
        if len(indices) == 0:
            ax.text(0.5, 0.5, 'No samples in this category', ha='center', va='center',
                   fontsize=14, transform=ax.transAxes)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')
            return
        
        # 限制样本数量
        if len(indices) > max_samples:
            indices = np.random.choice(indices, max_samples, replace=False)
        
        num_layers = self.num_layers
        
        # 对每层特征进行PCA降维到2D
        all_features_2d = []
        for layer_name, features in self.layer_features:
            features = self._preprocess_features(features)
            
            # PCA降维
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features)
            all_features_2d.append(features_2d)
        
        # 绘制样本轨迹
        cmap = plt.cm.Reds if 'False Positive' in title else plt.cm.Blues
        
        for idx_i, idx in enumerate(indices):
            trajectory_x = []
            trajectory_y = []
            
            for layer_idx in range(num_layers):
                # 使用层索引作为x坐标，PCA第一维作为y坐标
                trajectory_x.append(layer_idx)
                trajectory_y.append(all_features_2d[layer_idx][idx, 0])
            
            # 绘制轨迹线
            color = cmap(0.3 + 0.7 * idx_i / len(indices))
            ax.plot(trajectory_x, trajectory_y, color=color, alpha=0.5, linewidth=1)
            
            # 标记起点和终点
            ax.scatter(trajectory_x[0], trajectory_y[0], c='green', s=30, zorder=5, alpha=0.7)
            ax.scatter(trajectory_x[-1], trajectory_y[-1], c='red', s=30, marker='x', zorder=5, alpha=0.7)
        
        # 绘制层分隔线
        for i in range(num_layers):
            ax.axvline(x=i, color='gray', linestyle=':', alpha=0.5)
        
        # 设置标签
        ax.set_xticks(range(num_layers))
        short_names = [name.replace('Transformer ', 'T').replace('Layer ', 'L') 
                      for name in self.layer_names]
        ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
        ax.set_xlabel('Layer', fontsize=11)
        ax.set_ylabel('Feature Representation (PCA-1)', fontsize=11)
        ax.set_title(f'{title}\n(n={len(indices)} samples)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                      markersize=8, label='Input Layer'),
            plt.Line2D([0], [0], marker='x', color='red', markersize=8, 
                      markeredgewidth=2, label='Output (Misclassified)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    def _plot_detailed_analysis(self, save_dir):
        """绘制详细分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 错位度量变化曲线
        ax1 = axes[0, 0]
        x = range(self.num_layers)
        ax1.plot(x, self.layer_misalignment, 'o-', color='#e74c3c', linewidth=2, 
                markersize=8, label='Misalignment Score')
        ax1.fill_between(x, self.layer_misalignment, alpha=0.3, color='#e74c3c')
        ax1.set_xlabel('Layer', fontsize=12)
        ax1.set_ylabel('Misalignment Score', fontsize=12)
        ax1.set_title('Semantic Misalignment Across Layers', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        short_names = [name.replace('Transformer ', 'T').replace('Layer ', 'L') 
                      for name in self.layer_names]
        ax1.set_xticklabels(short_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 标记错位突增点
        for i in range(1, self.num_layers):
            if self.layer_misalignment[i] > self.layer_misalignment[i-1] * 1.2:
                ax1.annotate('⚠', (i, self.layer_misalignment[i]), 
                           fontsize=16, color='#f39c12', ha='center', va='bottom')
        
        # 2. 类间距离变化
        ax2 = axes[0, 1]
        ax2.bar(x, self.layer_class_distances, color='#3498db', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Layer', fontsize=12)
        ax2.set_ylabel('Class Distance', fontsize=12)
        ax2.set_title('Inter-class Distance Across Layers', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(short_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, v in enumerate(self.layer_class_distances):
            ax2.text(i, v + 0.01 * max(self.layer_class_distances), f'{v:.2f}', 
                    ha='center', va='bottom', fontsize=9)
        
        # 3. 类内方差变化
        ax3 = axes[1, 0]
        ax3.bar(x, self.layer_intra_variances, color='#9b59b6', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Layer', fontsize=12)
        ax3.set_ylabel('Intra-class Variance', fontsize=12)
        ax3.set_title('Intra-class Variance Across Layers', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(short_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, v in enumerate(self.layer_intra_variances):
            ax3.text(i, v + 0.01 * max(self.layer_intra_variances), f'{v:.2f}', 
                    ha='center', va='bottom', fontsize=9)
        
        # 4. 错位传播关系图
        ax4 = axes[1, 1]
        
        # 计算错位传播相关性
        misalignment_change = np.diff(self.layer_misalignment)
        distance_change = np.diff(self.layer_class_distances)
        
        ax4.scatter(misalignment_change, distance_change, c=range(len(misalignment_change)),
                   cmap='coolwarm', s=100, edgecolors='black', linewidth=1)
        
        # 添加层标签
        for i, (mx, dx) in enumerate(zip(misalignment_change, distance_change)):
            ax4.annotate(f'L{i+1}→L{i+2}', (mx, dx), fontsize=9, 
                        xytext=(5, 5), textcoords='offset points')
        
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Misalignment Change', fontsize=12)
        ax4.set_ylabel('Class Distance Change', fontsize=12)
        ax4.set_title('Misalignment vs Class Distance Change\n(Layer Transitions)', 
                     fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 添加象限标签
        ax4.text(0.95, 0.95, 'Misalignment↑\nSeparation↑', transform=ax4.transAxes,
                ha='right', va='top', fontsize=9, color='gray',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax4.text(0.05, 0.05, 'Misalignment↓\nSeparation↓', transform=ax4.transAxes,
                ha='left', va='bottom', fontsize=9, color='gray',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Detailed Semantic Misalignment Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, 'semantic_misalignment_detailed.png')
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close()
        print(f"详细分析图已保存到: {save_path}")
        
        # 保存详细分析数据到CSV
        short_names = [name.replace('Transformer ', 'T').replace('Layer ', 'L') 
                      for name in self.layer_names]
        detailed_df = pd.DataFrame({
            'Layer': self.layer_names,
            'Short_Name': short_names,
            'Misalignment_Score': self.layer_misalignment,
            'Class_Distance': self.layer_class_distances,
            'Intra_Class_Variance': self.layer_intra_variances
        })
        csv_path = os.path.join(save_dir, 'semantic_misalignment_detailed.csv')
        detailed_df.to_csv(csv_path, index=False)
        print(f"详细分析数据已保存到: {csv_path}")
        
        # 保存层间变化数据到CSV
        if self.num_layers > 1:
            misalignment_change = np.diff(self.layer_misalignment)
            distance_change = np.diff(self.layer_class_distances)
            transition_names = [f'L{i+1}→L{i+2}' for i in range(len(misalignment_change))]
            transition_df = pd.DataFrame({
                'Transition': transition_names,
                'Misalignment_Change': misalignment_change,
                'Class_Distance_Change': distance_change
            })
            transition_csv_path = os.path.join(save_dir, 'layer_transition_analysis.csv')
            transition_df.to_csv(transition_csv_path, index=False)
            print(f"层间变化数据已保存到: {transition_csv_path}")
    
    def generate_report(self, save_dir):
        """生成分析报告"""
        report = []
        report.append("=" * 60)
        report.append("语义错位传播分析报告")
        report.append("Semantic Misalignment Propagation Analysis Report")
        report.append("=" * 60)
        report.append("")
        
        # 基本统计
        report.append("【基本统计】")
        report.append(f"总样本数: {len(self.labels)}")
        report.append(f"正确分类: {np.sum(self.correct_mask)} ({np.sum(self.correct_mask)/len(self.labels)*100:.2f}%)")
        report.append(f"错误分类: {np.sum(self.incorrect_mask)} ({np.sum(self.incorrect_mask)/len(self.labels)*100:.2f}%)")
        report.append(f"  - 误报 (Normal→Attack): {np.sum(self.false_positive_mask)}")
        report.append(f"  - 漏报 (Attack→Normal): {np.sum(self.false_negative_mask)}")
        report.append("")
        
        # 层级错位分析
        report.append("【层级错位分析】")
        max_misalignment_layer = np.argmax(self.layer_misalignment)
        report.append(f"最高错位层: {self.layer_names[max_misalignment_layer]} (错位度: {self.layer_misalignment[max_misalignment_layer]:.4f})")
        
        # 找出错位突增层
        for i in range(1, self.num_layers):
            if self.layer_misalignment[i] > self.layer_misalignment[i-1] * 1.2:
                report.append(f"⚠ 错位突增层: {self.layer_names[i]} (增幅: {(self.layer_misalignment[i]/self.layer_misalignment[i-1]-1)*100:.1f}%)")
        report.append("")
        
        # 各层详细数据
        report.append("【各层详细数据】")
        report.append(f"{'层名称':<20} {'错位度':<12} {'类间距离':<12} {'类内方差':<12}")
        report.append("-" * 56)
        for i, name in enumerate(self.layer_names):
            short_name = name[:18] if len(name) > 18 else name
            report.append(f"{short_name:<20} {self.layer_misalignment[i]:<12.4f} {self.layer_class_distances[i]:<12.4f} {self.layer_intra_variances[i]:<12.4f}")
        report.append("")
        
        # 结论
        report.append("【分析结论】")
        if max_misalignment_layer < self.num_layers // 2:
            report.append("• 语义错位主要发生在模型前半部分，建议加强输入层特征提取能力")
        else:
            report.append("• 语义错位主要发生在模型后半部分，建议优化高层特征融合策略")
        
        if np.sum(self.false_positive_mask) > np.sum(self.false_negative_mask):
            report.append("• 误报率高于漏报率，模型倾向于将Normal误判为Attack")
        else:
            report.append("• 漏报率高于误报率，模型倾向于将Attack漏报为Normal")
        
        report.append("=" * 60)
        
        # 保存报告
        report_text = "\n".join(report)
        report_path = os.path.join(save_dir, 'misalignment_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"分析报告已保存到: {report_path}")
        
        return report_text


def analyze_transformer_misalignment(model, features, labels, save_dir, model_name="Transformer"):
    """
    分析Transformer模型的语义错位
    
    Args:
        model: 训练好的Transformer模型
        features: 输入特征
        labels: 真实标签
        save_dir: 保存目录
        model_name: 模型名称
    
    Returns:
        analyzer: SemanticMisalignmentAnalyzer实例
    """
    import tensorflow as tf
    
    print("\n" + "=" * 60)
    print(f"开始{model_name}语义错位分析...")
    print("=" * 60)
    
    # 获取各层特征
    print("[1] 提取各层特征...")
    layer_features = model.get_layer_features(tf.constant(features))
    
    # 获取预测结果
    print("[2] 获取模型预测...")
    predictions = model.predict(features)
    predictions = np.argmax(predictions, axis=1)
    
    # 创建分析器
    print("[3] 创建语义错位分析器...")
    analyzer = SemanticMisalignmentAnalyzer(layer_features, labels, predictions)
    
    # 生成可视化
    print("[4] 生成语义错位传播图...")
    analyzer.plot_misalignment_propagation_graph(save_dir)
    
    # 生成报告
    print("[5] 生成分析报告...")
    report = analyzer.generate_report(save_dir)
    print(report)
    
    print("\n" + "=" * 60)
    print(f"语义错位分析完成！结果保存在: {save_dir}")
    print("=" * 60)
    
    return analyzer


if __name__ == "__main__":
    print("语义错位传播图模块")
    print("请在训练好Transformer模型后调用 analyze_transformer_misalignment() 函数")
