"""
Visualization utilities for MEDAL-Lite
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import os
import logging

# Try to import scipy for KDE, fallback gracefully if not available
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")


def _configure_matplotlib_chinese_fonts():
    """Configure matplotlib to use Chinese fonts and suppress warnings if unavailable."""
    import warnings
    # Suppress font warnings globally if no Chinese font is available
    warnings.filterwarnings('ignore', category=UserWarning, 
                          message='.*Glyph.*missing from font.*')
    
    try:
        preferred = [
            'Noto Sans CJK SC',
            'Noto Sans CJK TC',
            'Noto Sans CJK JP',
            'Noto Sans SC',
            'Noto Sans TC',
            'Source Han Sans SC',
            'Source Han Sans CN',
            'WenQuanYi Zen Hei',
            'WenQuanYi Micro Hei',
            'Microsoft YaHei',
            'SimHei'
        ]
        available = {f.name for f in font_manager.fontManager.ttflist}
        chosen = None
        for name in preferred:
            if name in available:
                chosen = name
                break

        if chosen is not None:
            current = mpl.rcParams.get('font.sans-serif', [])
            if not isinstance(current, list):
                current = [current]
            mpl.rcParams['font.family'] = 'sans-serif'
            mpl.rcParams['font.sans-serif'] = [chosen] + [f for f in current if f != chosen]

        mpl.rcParams['axes.unicode_minus'] = False
    except Exception:
        # If font configuration fails, warnings are already suppressed above
        pass


_configure_matplotlib_chinese_fonts()


def plot_feature_space(features, labels, save_path, title="Feature Space", method='tsne'):
    """
    Plot 2D visualization of feature space
    
    Args:
        features: (N, d) - feature vectors
        labels: (N,) - labels (0=benign, 1=malicious)
        save_path: str - path to save figure
        title: str - plot title
        method: str - 'tsne' or 'pca'
    """
    logger.info(f"Generating feature space visualization using {method.upper()}...")
    
    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    else:  # pca
        reducer = PCA(n_components=2, random_state=42)
    
    features_2d = reducer.fit_transform(features)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Plot benign samples (blue)
    benign_mask = labels == 0
    plt.scatter(features_2d[benign_mask, 0], features_2d[benign_mask, 1],
                c='blue', label='Benign', alpha=0.6, s=30, edgecolors='k', linewidths=0.5)
    
    # Plot malicious samples (red)
    malicious_mask = labels == 1
    plt.scatter(features_2d[malicious_mask, 0], features_2d[malicious_mask, 1],
                c='red', label='Malicious', alpha=0.6, s=30, edgecolors='k', linewidths=0.5)
    
    plt.xlabel(f'{method.upper()} Component 1', fontsize=12)
    plt.ylabel(f'{method.upper()} Component 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Feature space plot saved to {save_path}")


def plot_noise_correction_comparison(original_labels, noisy_labels, corrected_labels, action_mask, save_path):
    """
    Visualize label noise correction results
    
    Args:
        original_labels: (N,) - ground truth labels
        noisy_labels: (N,) - noisy labels
        corrected_labels: (N,) - corrected labels
        action_mask: (N,) - actions taken (0=keep, 1=flip, 2=drop, 3=reweight)
        save_path: str - path to save figure
    """
    logger.info("Generating noise correction comparison plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Noisy vs Original
    ax = axes[0, 0]
    errors_noisy = (noisy_labels != original_labels)
    accuracy_noisy = 100 * (1 - errors_noisy.mean())
    
    indices = np.arange(len(original_labels))
    ax.scatter(indices[~errors_noisy], original_labels[~errors_noisy], 
               c='green', label='Correct', alpha=0.5, s=10)
    ax.scatter(indices[errors_noisy], original_labels[errors_noisy], 
               c='red', label='Incorrect', alpha=0.7, s=20, marker='x')
    ax.set_title(f'Noisy Labels (Accuracy: {accuracy_noisy:.1f}%)', fontweight='bold')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Label')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Corrected vs Original (excluding dropped)
    ax = axes[0, 1]
    keep_mask = action_mask != 2
    errors_corrected = (corrected_labels[keep_mask] != original_labels[keep_mask])
    accuracy_corrected = 100 * (1 - errors_corrected.mean())
    
    indices_keep = indices[keep_mask]
    ax.scatter(indices_keep[~errors_corrected], original_labels[keep_mask][~errors_corrected], 
               c='green', label='Correct', alpha=0.5, s=10)
    ax.scatter(indices_keep[errors_corrected], original_labels[keep_mask][errors_corrected], 
               c='red', label='Incorrect', alpha=0.7, s=20, marker='x')
    ax.set_title(f'Corrected Labels (Accuracy: {accuracy_corrected:.1f}%)', fontweight='bold')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Label')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Action distribution
    ax = axes[1, 0]
    action_names = ['Keep', 'Flip', 'Drop', 'Reweight']
    action_counts = [(action_mask == i).sum() for i in range(4)]
    colors = ['green', 'orange', 'red', 'blue']
    
    bars = ax.bar(action_names, action_counts, color=colors, alpha=0.7, edgecolor='black')
    ax.set_title('Action Distribution', fontweight='bold')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for bar, count in zip(bars, action_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({100*count/len(action_mask):.1f}%)',
                ha='center', va='bottom', fontsize=10)
    
    # 4. Confusion matrix: noisy vs corrected
    ax = axes[1, 1]
    
    metrics_data = {
        'Initial Noise Rate': [100 * errors_noisy.mean()],
        'Corrected Error Rate': [100 * errors_corrected.mean()],
        'Dropped Samples': [100 * (action_mask == 2).sum() / len(action_mask)],
        'Improvement': [100 * (errors_noisy.mean() - errors_corrected.mean())]
    }
    
    y_pos = np.arange(len(metrics_data))
    values = [v[0] for v in metrics_data.values()]
    colors_bar = ['red', 'orange', 'gray', 'green']
    
    bars = ax.barh(y_pos, values, color=colors_bar, alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(list(metrics_data.keys()))
    ax.set_xlabel('Percentage (%)')
    ax.set_title('Correction Metrics', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, value in zip(bars, values):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{value:.1f}%',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Noise correction comparison plot saved to {save_path}")


def plot_training_history(history, save_path):
    """
    Plot training history (loss and metrics over epochs)
    
    Args:
        history: dict with keys like 'train_loss', 'val_loss', 'train_f1', 'val_f1', etc.
        save_path: str - path to save figure
    """
    logger.info("Generating training history plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss over Epochs', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # F1 Score
    ax = axes[0, 1]
    if 'train_f1' in history:
        ax.plot(epochs, history['train_f1'], 'b-', label='Train F1', linewidth=2)
    if 'val_f1' in history:
        ax.plot(epochs, history['val_f1'], 'r-', label='Val F1', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score over Epochs', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Dynamic weights
    ax = axes[1, 0]
    if 'lambda_orth' in history:
        ax.plot(epochs, history['lambda_orth'], 'g-', label='λ_orth', linewidth=2)
    if 'lambda_con' in history:
        ax.plot(epochs, history['lambda_con'], 'm-', label='λ_con', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weight')
    ax.set_title('Dynamic Loss Weights', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy
    ax = axes[1, 1]
    if 'train_acc' in history:
        ax.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    if 'val_acc' in history:
        ax.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy over Epochs', fontweight='bold')
    if ('train_acc' in history) or ('val_acc' in history):
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Training history plot saved to {save_path}")


def plot_confusion_matrix(cm, class_names, save_path, title='Confusion Matrix'):
    """
    Plot confusion matrix
    
    Args:
        cm: (2, 2) confusion matrix
        class_names: list of class names
        save_path: str - path to save figure
        title: str - plot title
    """
    plt.figure(figsize=(8, 6))
    
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage'})
    
    # Add counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j + 0.5, i + 0.7, f'({cm[i, j]})',
                    ha='center', va='center', fontsize=10, color='gray')
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved to {save_path}")


def plot_roc_curve(y_true, y_prob, save_path, title='ROC Curve'):
    """
    Plot ROC curve
    
    Args:
        y_true: (N,) - true binary labels (0=benign, 1=malicious)
        y_prob: (N, 2) - predicted probabilities for both classes
        save_path: str - path to save figure
        title: str - plot title
    """
    logger.info("Generating ROC curve...")
    
    # Extract probabilities for malicious class (class 1)
    if y_prob.shape[1] == 2:
        y_scores = y_prob[:, 1]  # Probability of malicious class
    else:
        y_scores = y_prob  # Already 1D probabilities
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier (AUC = 0.5000)')
    
    # Formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add text box with key metrics
    textstr = f'AUC = {roc_auc:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.6, 0.2, textstr, fontsize=12, verticalalignment='top', 
             bbox=props, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"ROC curve saved to {save_path} (AUC = {roc_auc:.4f})")


def plot_probability_distribution(y_true, y_prob, save_path, title='Probability Distribution', threshold=None):
    """
    Plot probability distribution for benign and malicious samples
    
    Args:
        y_true: (N,) - true binary labels (0=benign, 1=malicious)
        y_prob: (N, 2) - predicted probabilities for both classes
        save_path: str - path to save figure
        title: str - plot title
        threshold: float - decision threshold to display as vertical line
    """
    logger.info("Generating probability distribution plot...")
    
    # Extract probabilities for malicious class (class 1)
    if y_prob.shape[1] == 2:
        malicious_probs = y_prob[:, 1]  # Probability of malicious class
    else:
        malicious_probs = y_prob  # Already 1D probabilities
    
    # Separate by true labels
    benign_mask = y_true == 0
    malicious_mask = y_true == 1
    
    benign_probs = malicious_probs[benign_mask]
    malicious_probs_true = malicious_probs[malicious_mask]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Histogram with density curve
    ax = axes[0]
    
    # Histogram for benign samples
    n_bins = 50
    counts_benign, bins_benign, patches_benign = ax.hist(
        benign_probs, bins=n_bins, alpha=0.6, color='blue', 
        label=f'Benign (n={len(benign_probs)})', density=True, edgecolor='black', linewidth=0.5
    )
    
    # Histogram for malicious samples
    counts_malicious, bins_malicious, patches_malicious = ax.hist(
        malicious_probs_true, bins=n_bins, alpha=0.6, color='red',
        label=f'Malicious (n={len(malicious_probs_true)})', density=True, edgecolor='black', linewidth=0.5
    )
    
    # Add KDE curves
    if HAS_SCIPY:
        try:
            # KDE for benign
            kde_benign = stats.gaussian_kde(benign_probs)
            x_benign = np.linspace(benign_probs.min(), benign_probs.max(), 200)
            ax.plot(x_benign, kde_benign(x_benign), 'b-', linewidth=2.5, label='Benign KDE', alpha=0.8)
            
            # KDE for malicious
            kde_malicious = stats.gaussian_kde(malicious_probs_true)
            x_malicious = np.linspace(malicious_probs_true.min(), malicious_probs_true.max(), 200)
            ax.plot(x_malicious, kde_malicious(x_malicious), 'r-', linewidth=2.5, label='Malicious KDE', alpha=0.8)
        except Exception as e:
            # Fallback if KDE calculation fails
            logger.warning(f"Could not compute KDE curves: {e}")
    
    # Add threshold line if provided
    if threshold is not None:
        ax.axvline(x=threshold, color='green', linestyle='--', linewidth=2, 
                  label=f'Decision Threshold ({threshold:.2f})', alpha=0.8)
    
    ax.set_xlabel('Predicted Probability (Malicious Class)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{title} - Histogram with KDE', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim([0, 1])
    
    # Add statistics text box
    stats_text = (
        f'Benign: μ={benign_probs.mean():.3f}, σ={benign_probs.std():.3f}, '
        f'median={np.median(benign_probs):.3f}\n'
        f'Malicious: μ={malicious_probs_true.mean():.3f}, σ={malicious_probs_true.std():.3f}, '
        f'median={np.median(malicious_probs_true):.3f}'
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props, family='monospace')
    
    # Plot 2: Cumulative Distribution Function (CDF)
    ax = axes[1]
    
    # Sort probabilities for CDF
    benign_sorted = np.sort(benign_probs)
    malicious_sorted = np.sort(malicious_probs_true)
    
    # Calculate CDF
    benign_cdf = np.arange(1, len(benign_sorted) + 1) / len(benign_sorted)
    malicious_cdf = np.arange(1, len(malicious_sorted) + 1) / len(malicious_sorted)
    
    # Plot CDF
    ax.plot(benign_sorted, benign_cdf, 'b-', linewidth=2.5, label='Benign CDF', alpha=0.8)
    ax.plot(malicious_sorted, malicious_cdf, 'r-', linewidth=2.5, label='Malicious CDF', alpha=0.8)
    
    # Add threshold line if provided
    if threshold is not None:
        ax.axvline(x=threshold, color='green', linestyle='--', linewidth=2, 
                  label=f'Decision Threshold ({threshold:.2f})', alpha=0.8)
        # Add horizontal line at threshold intersection
        benign_at_threshold = (benign_sorted <= threshold).sum() / len(benign_sorted)
        malicious_at_threshold = (malicious_sorted <= threshold).sum() / len(malicious_sorted)
        ax.plot([threshold, threshold], [0, max(benign_at_threshold, malicious_at_threshold)], 
                'g--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Predicted Probability (Malicious Class)', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title(f'{title} - Cumulative Distribution Function', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Probability distribution plot saved to {save_path}")


def plot_real_vs_synthetic_comparison(X_real, X_synthetic, y_real, y_synthetic, save_path, 
                                       title='Real vs Synthetic Data Comparison', method='tsne'):
    """
    Plot comparison between real and synthetic data in 2D feature space
    
    用于评估生成质量的可视化：
    - 蓝色: 真实良性样本
    - 红色: 真实恶意样本
    - 浅蓝色: 合成良性样本
    - 浅红色: 合成恶意样本
    
    好的生成结果: 合成样本应该覆盖在真实样本之上，像影子一样
    坏的生成结果: 合成样本聚成一团，与真实样本分离 (Mode Collapse)
    
    Args:
        X_real: (N_real, L, 5) - real samples (sequences)
        X_synthetic: (N_syn, L, 5) - synthetic samples (sequences)
        y_real: (N_real,) - real labels
        y_synthetic: (N_syn,) - synthetic labels
        save_path: str - path to save figure
        title: str - plot title
        method: str - 'tsne' or 'pca'
    """
    logger.info(f"Generating real vs synthetic comparison using {method.upper()}...")
    
    # Flatten sequences to packet-level features (average over sequence)
    X_real_flat = X_real.mean(axis=1)  # (N_real, 5)
    X_synthetic_flat = X_synthetic.mean(axis=1)  # (N_syn, 5)
    
    # Combine for joint dimensionality reduction
    X_combined = np.concatenate([X_real_flat, X_synthetic_flat], axis=0)
    
    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_combined)//4), max_iter=1000)
    else:  # pca
        reducer = PCA(n_components=2, random_state=42)
    
    X_2d = reducer.fit_transform(X_combined)
    
    # Split back
    n_real = len(X_real)
    X_real_2d = X_2d[:n_real]
    X_synthetic_2d = X_2d[n_real:]
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot 1: Real data only
    ax = axes[0]
    benign_mask_real = y_real == 0
    malicious_mask_real = y_real == 1
    
    ax.scatter(X_real_2d[benign_mask_real, 0], X_real_2d[benign_mask_real, 1],
              c='blue', label='Real Benign', alpha=0.7, s=50, edgecolors='k', linewidths=0.5)
    ax.scatter(X_real_2d[malicious_mask_real, 0], X_real_2d[malicious_mask_real, 1],
              c='red', label='Real Malicious', alpha=0.7, s=50, edgecolors='k', linewidths=0.5)
    
    ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
    ax.set_title('Real Data Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Real + Synthetic overlay
    ax = axes[1]
    
    # Plot synthetic first (background)
    benign_mask_syn = y_synthetic == 0
    malicious_mask_syn = y_synthetic == 1
    
    ax.scatter(X_synthetic_2d[benign_mask_syn, 0], X_synthetic_2d[benign_mask_syn, 1],
              c='lightblue', label='Synthetic Benign', alpha=0.4, s=30, marker='o')
    ax.scatter(X_synthetic_2d[malicious_mask_syn, 0], X_synthetic_2d[malicious_mask_syn, 1],
              c='lightcoral', label='Synthetic Malicious', alpha=0.4, s=30, marker='o')
    
    # Plot real on top (foreground)
    ax.scatter(X_real_2d[benign_mask_real, 0], X_real_2d[benign_mask_real, 1],
              c='blue', label='Real Benign', alpha=0.7, s=50, edgecolors='k', linewidths=0.5)
    ax.scatter(X_real_2d[malicious_mask_real, 0], X_real_2d[malicious_mask_real, 1],
              c='red', label='Real Malicious', alpha=0.7, s=50, edgecolors='k', linewidths=0.5)
    
    ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
    ax.set_title('Real vs Synthetic Overlay', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = (
        f'Real: {len(X_real)} samples (Benign={benign_mask_real.sum()}, Malicious={malicious_mask_real.sum()})\n'
        f'Synthetic: {len(X_synthetic)} samples (Benign={benign_mask_syn.sum()}, Malicious={malicious_mask_syn.sum()})'
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Real vs synthetic comparison plot saved to {save_path}")
