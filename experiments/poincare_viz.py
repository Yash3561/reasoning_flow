import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

def project_to_poincare(h, scale=0.85):
    """Project hidden states to 2D Poincare disk."""
    # PCA to 2D
    pca = PCA(n_components=2)
    z = pca.fit_transform(h)
    # Normalize to disk
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    max_norm = np.percentile(norms, 95)
    r = np.tanh(norms / max_norm) * scale
    z_disk = z / np.maximum(norms, 1e-8) * r
    return z_disk

def draw_disk(ax):
    circle = plt.Circle((0, 0), 1.0, fill=False, color='gray', linewidth=1.5)
    ax.add_patch(circle)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')

def make_poincare_grid(truth_h, lie_h, model_name, out_dir, n_layers=None):
    os.makedirs(out_dir, exist_ok=True)
    if n_layers is None:
        n_layers = truth_h.shape[1]
    
    # Show layers 1-32 (skip layer 0 embedding)
    layers_to_show = list(range(1, n_layers))
    n_show = len(layers_to_show)
    cols = 8
    rows = (n_show + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.5, rows*2.5))
    fig.suptitle(f'Poincare Disk — Truth vs Lie Hidden States\n{model_name}', fontsize=14, y=1.01)
    axes = axes.flatten()
    
    for idx, layer in enumerate(layers_to_show):
        ax = axes[idx]
        draw_disk(ax)
        
        t = truth_h[:, layer, :]
        l = lie_h[:, layer, :]
        combined = np.vstack([t, l])
        
        z = project_to_poincare(combined)
        z_truth = z[:len(t)]
        z_lie   = z[len(t):]
        
        ax.scatter(z_truth[:, 0], z_truth[:, 1], c='#2ecc71', s=8, alpha=0.7, label='Truth')
        ax.scatter(z_lie[:, 0],   z_lie[:, 1],   c='#e74c3c', s=8, alpha=0.7, label='Lie')
        ax.set_title(f'L{layer}', fontsize=9, pad=2)
    
    # Hide unused axes
    for idx in range(len(layers_to_show), len(axes)):
        axes[idx].axis('off')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=8, label='Truth'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=8, label='Lie'),
    ]
    fig.legend(handles=legend_elements, loc='lower right', fontsize=10, ncol=2)
    
    plt.tight_layout()
    path = f'{out_dir}/poincare_grid.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved → {path}")

# Run for both models
print("Generating Llama Poincare disk visualizations...")
truth_llama = np.load('results/layerwise_200/llama_final/truth_all_layers.npy')
lie_llama   = np.load('results/layerwise_200/llama_final/lie_all_layers.npy')
make_poincare_grid(truth_llama, lie_llama, 'Llama-3.1-8B', 'results/poincare/llama')

print("Generating Qwen Poincare disk visualizations...")
truth_qwen = np.load('results/layerwise_200/qwen_final/truth_all_layers.npy')
lie_qwen   = np.load('results/layerwise_200/qwen_final/lie_all_layers.npy')
make_poincare_grid(truth_qwen, lie_qwen, 'Qwen2.5-7B', 'results/poincare/qwen')

print("Done!")
