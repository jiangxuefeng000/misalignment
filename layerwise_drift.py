import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from matplotlib.lines import Line2D

def load_layers_with_labels(base_dir, epoch, layers, label_path=None):
    reps = []
    for l in layers:
        path = os.path.join(base_dir, f"epoch_{epoch}_layer_{l}.npz")
        data = np.load(path)
        reps.append(data['embeddings'])

    labels = None
    if label_path and os.path.exists(label_path):
        labels = np.load(label_path)['labels']
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
    reps, labels = load_layers_with_labels(hidden_dir, epoch, layers, label_path)
    L, N, D = reps.shape
    
    base_layer = 0
    pca = PCA(n_components=2)

    reps_2d = np.zeros((L, min(N, max_points), 2))
    for l in range(L):
        reps_2d[l] = pca.transform(reps[l][:max_points])

    if labels is not None:
        unique_labels = np.unique(labels[:max_points])
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
        point_colors = [label_to_color[label] for label in labels[:max_points]]
    else:
        point_colors = plt.cm.rainbow(np.linspace(0, 1, min(N, max_points)))
    for i in range(min(N, max_points)):
        xs = reps_2d[:, i, 0]
        ys = reps_2d[:, i, 1]
        

        ax1.plot(xs, ys, alpha=0.3, color=point_colors[i], linewidth=1)

        ax1.scatter(xs[0], ys[0], color=point_colors[i], s=30, alpha=0.7)
        ax1.scatter(xs[-1], ys[-1], color=point_colors[i], s=50, alpha=0.9, 
                   marker='s', edgecolors='black')

        ax1.text(xs[0], ys[0], 'L0', fontsize=8, ha='center', va='center')
        ax1.text(xs[-1], ys[-1], f'L{layers[-1]}', fontsize=8, ha='center', va='center')
    
    ax1.set_title(f'Layer-wise Representation Trajectories\n(Epoch {epoch}, N={min(N, max_points)})')
    ax1.set_xlabel('PCA Component 1')
    ax1.set_ylabel('PCA Component 2')
    ax1.grid(True, alpha=0.3)
    
    print("\n=== Layer Drift Analysis ===")
    for i in range(L-1):
        layer_i = reps_2d[i]
        layer_j = reps_2d[i+1]
        distances = np.linalg.norm(layer_j - layer_i, axis=1)
        avg_distance = np.mean(distances)
        print(f"L{layers[i]} → L{layers[i+1]}: Avg Euclidean Distance = {avg_distance:.4f}")
    
    plt.show()

if __name__ == "__main__":
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
        label_path="path/to/labels.npz", 
        attack_types=ATTACK_TYPES 

    )
