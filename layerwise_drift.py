import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from matplotlib.lines import Line2D

def load_layers_with_labels(base_dir, epoch, layers, label_path=None):
    """
    加载隐藏层表示和对应的标签
    """
    reps = []
    for l in layers:
        path = os.path.join(base_dir, f"epoch_{epoch}_layer_{l}.npz")
        data = np.load(path)
        reps.append(data['embeddings'])
    
    # 加载标签（如果有）
    labels = None
    if label_path and os.path.exists(label_path):
        labels = np.load(label_path)['labels']
        # 确保标签数量匹配
        if len(labels) != reps[0].shape[0]:
            print(f"Warning: Labels shape {len(labels)} != embeddings shape {reps[0].shape[0]}")
    
    return np.stack(reps, axis=0), labels

def plot_layerwise_drift_improved(
    hidden_dir="hidden_states",
    epoch=4,
    layers=[0, 1, 2, 3],
    max_points=50,
    label_path=None,
    attack_types=None
):
    """
    改进的跨层语义漂移轨迹图
    """
    # 加载数据
    reps, labels = load_layers_with_labels(hidden_dir, epoch, layers, label_path)
    L, N, D = reps.shape
    
    # 处理NaN值
    reps = np.nan_to_num(reps, nan=0.0)
    
    # 使用第一层（通常是输入层）作为参考系进行PCA
    # 这样可以观察相同样本在不同层的投影变化
    base_layer = 0
    pca = PCA(n_components=2)
    
    # 使用第一层的数据拟合PCA
    base_reps = reps[base_layer]
    pca.fit(base_reps[:max_points])
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    
    # 对所有层应用相同的PCA变换
    reps_2d = np.zeros((L, min(N, max_points), 2))
    for l in range(L):
        reps_2d[l] = pca.transform(reps[l][:max_points])
    
    # 设置颜色和样式
    if labels is not None:
        # 如果有标签，根据标签设置颜色
        unique_labels = np.unique(labels[:max_points])
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
        point_colors = [label_to_color[label] for label in labels[:max_points]]
    else:
        # 没有标签，使用彩虹色
        point_colors = plt.cm.rainbow(np.linspace(0, 1, min(N, max_points)))
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 子图1：轨迹图
    ax1 = axes[0]
    
    # 绘制每个样本的跨层轨迹
    for i in range(min(N, max_points)):
        xs = reps_2d[:, i, 0]
        ys = reps_2d[:, i, 1]
        
        # 绘制轨迹线
        ax1.plot(xs, ys, alpha=0.3, color=point_colors[i], linewidth=1)
        
        # 标记起始层（L0）和结束层（L3）
        ax1.scatter(xs[0], ys[0], color=point_colors[i], s=30, alpha=0.7)
        ax1.scatter(xs[-1], ys[-1], color=point_colors[i], s=50, alpha=0.9, 
                   marker='s', edgecolors='black')
        
        # 在起点和终点添加层标记
        ax1.text(xs[0], ys[0], 'L0', fontsize=8, ha='center', va='center')
        ax1.text(xs[-1], ys[-1], f'L{layers[-1]}', fontsize=8, ha='center', va='center')
    
    ax1.set_title(f'Layer-wise Representation Trajectories\n(Epoch {epoch}, N={min(N, max_points)})')
    ax1.set_xlabel('PCA Component 1')
    ax1.set_ylabel('PCA Component 2')
    ax1.grid(True, alpha=0.3)
    
    # 子图2：层间距离热力图
    ax2 = axes[1]
    
    # 计算层间表示的平均余弦距离
    cos_dist_matrix = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            # 计算余弦相似度
            from sklearn.metrics.pairwise import cosine_similarity
            sim = cosine_similarity(reps[i][:max_points], reps[j][:max_points])
            # 相似度转换为距离 (1 - 相似度)
            avg_dist = 1 - np.mean(np.diag(sim))
            cos_dist_matrix[i, j] = avg_dist
    
    # 绘制热力图
    im = ax2.imshow(cos_dist_matrix, cmap='YlOrRd')
    ax2.set_title('Inter-layer Cosine Distance Matrix')
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Layer Index')
    
    # 添加数值标签
    for i in range(L):
        for j in range(L):
            ax2.text(j, i, f'{cos_dist_matrix[i, j]:.3f}', 
                    ha='center', va='center', color='black' if cos_dist_matrix[i, j] < 0.5 else 'white')
    
    # 设置刻度标签
    layer_labels = [f'L{l}' for l in layers]
    ax2.set_xticks(range(L))
    ax2.set_yticks(range(L))
    ax2.set_xticklabels(layer_labels)
    ax2.set_yticklabels(layer_labels)
    
    # 添加颜色条
    plt.colorbar(im, ax=ax2, label='Cosine Distance')
    
    # 添加图例（如果有标签）
    if labels is not None and attack_types is not None:
        legend_elements = []
        for label, color in label_to_color.items():
            if label < len(attack_types):
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=color, markersize=10, 
                          label=attack_types[label])
                )
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    # 保存图表
    save_path = f"improved_layerwise_drift_epoch_{epoch}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Improved plot saved to: {save_path}")
    
    # 额外分析：计算平均漂移距离
    print("\n=== Layer Drift Analysis ===")
    for i in range(L-1):
        layer_i = reps_2d[i]
        layer_j = reps_2d[i+1]
        distances = np.linalg.norm(layer_j - layer_i, axis=1)
        avg_distance = np.mean(distances)
        print(f"L{layers[i]} → L{layers[i+1]}: Avg Euclidean Distance = {avg_distance:.4f}")
    
    plt.show()

if __name__ == "__main__":
    # 假设的攻击类型标签（根据你的实际数据集调整）
    ATTACK_TYPES = [
        "Normal",
        "DoS", 
        "Probe",
        "R2L",
        "U2R"
    ]
    
    plot_layerwise_drift_improved(
        hidden_dir="hidden_states",
        epoch=4,
        layers=[0, 1, 2, 3],
        max_points=50,
        label_path="path/to/labels.npz",  # 可选：标签文件路径
        attack_types=ATTACK_TYPES  # 可选：攻击类型名称
    )