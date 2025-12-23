"""
Analysis script for label correction results
Demonstrates how to analyze the generated CSV and NPZ files
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

sns.set_style("whitegrid")


def load_results(analysis_dir="output/label_correction/analysis"):
    """Load analysis results"""
    csv_path = os.path.join(analysis_dir, "documents", "sample_analysis.csv")
    npz_path = os.path.join(analysis_dir, "correction_results.npz")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    
    df = pd.read_csv(csv_path)
    data = np.load(npz_path)
    
    return df, data


def print_basic_statistics(df):
    """Print basic statistics"""
    print("="*70)
    print("BASIC STATISTICS")
    print("="*70)
    
    print(f"\n📊 Dataset Overview:")
    print(f"  Total samples:    {len(df)}")
    print(f"  Benign samples:   {(df['True_Label'] == 'Benign').sum()} ({(df['True_Label'] == 'Benign').mean()*100:.1f}%)")
    print(f"  Malicious samples: {(df['True_Label'] == 'Malicious').sum()} ({(df['True_Label'] == 'Malicious').mean()*100:.1f}%)")
    
    print(f"\n🔀 Noise Statistics:")
    n_noise = df['Is_Noise'].sum()
    noise_rate = n_noise / len(df) * 100
    print(f"  Noisy samples:    {n_noise} ({noise_rate:.1f}%)")
    print(f"  Clean samples:    {len(df) - n_noise} ({100 - noise_rate:.1f}%)")
    
    print(f"\n🔧 Correction Actions:")
    action_counts = df['Action'].value_counts()
    for action, count in action_counts.items():
        print(f"  {action:12s}: {count:5d} ({count/len(df)*100:.1f}%)")
    
    print(f"\n✓ Correction Accuracy:")
    kept_samples = df[df['Action'] != 'Drop']
    if len(kept_samples) > 0:
        n_correct = (kept_samples['Correction_Correct'] == 'Correct').sum()
        n_total = len(kept_samples)
        accuracy = n_correct / n_total * 100
        print(f"  Correct:   {n_correct}/{n_total} ({accuracy:.2f}%)")
        print(f"  Incorrect: {n_total - n_correct}/{n_total} ({100-accuracy:.2f}%)")


def analyze_by_action(df):
    """Analyze correction performance by action type"""
    print("\n" + "="*70)
    print("ANALYSIS BY ACTION TYPE")
    print("="*70)
    
    for action in ['Keep', 'Flip', 'Reweight']:
        action_df = df[df['Action'] == action]
        
        if len(action_df) == 0:
            continue
        
        print(f"\n{'='*70}")
        print(f"Action: {action}")
        print(f"{'='*70}")
        
        n_correct = (action_df['Correction_Correct'] == 'Correct').sum()
        n_total = len(action_df)
        accuracy = n_correct / n_total * 100 if n_total > 0 else 0
        
        print(f"  Total samples:      {n_total}")
        print(f"  Correct:            {n_correct} ({accuracy:.2f}%)")
        print(f"  Incorrect:          {n_total - n_correct} ({100-accuracy:.2f}%)")
        
        # True label distribution
        print(f"\n  True label distribution:")
        for label in ['Benign', 'Malicious']:
            n = (action_df['True_Label'] == label).sum()
            print(f"    {label:12s}: {n} ({n/n_total*100:.1f}%)")
        
        # Noise distribution
        n_noise = action_df['Is_Noise'].sum()
        print(f"\n  Noise distribution:")
        print(f"    Noisy:  {n_noise} ({n_noise/n_total*100:.1f}%)")
        print(f"    Clean:  {n_total - n_noise} ({(n_total-n_noise)/n_total*100:.1f}%)")


def analyze_cl_performance(df):
    """Analyze Confident Learning performance"""
    print("\n" + "="*70)
    print("CONFIDENT LEARNING (CL) PERFORMANCE")
    print("="*70)
    
    # CL suspected noise vs actual noise
    tp = ((df['CL_Suspected_Noise'] == True) & (df['Is_Noise'] == True)).sum()
    fp = ((df['CL_Suspected_Noise'] == True) & (df['Is_Noise'] == False)).sum()
    fn = ((df['CL_Suspected_Noise'] == False) & (df['Is_Noise'] == True)).sum()
    tn = ((df['CL_Suspected_Noise'] == False) & (df['Is_Noise'] == False)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(df)
    
    print(f"\n  Confusion Matrix (CL Suspected Noise vs Actual Noise):")
    print(f"                 Actual Noise  Actual Clean")
    print(f"  CL Suspected:  {tp:5d}         {fp:5d}")
    print(f"  CL Clean:      {fn:5d}         {tn:5d}")
    
    print(f"\n  Metrics:")
    print(f"    Precision: {precision*100:.2f}%")
    print(f"    Recall:    {recall*100:.2f}%")
    print(f"    F1-Score:  {f1*100:.2f}%")
    print(f"    Accuracy:  {accuracy*100:.2f}%")


def analyze_made_performance(df):
    """Analyze MADE density estimation performance"""
    print("\n" + "="*70)
    print("MADE DENSITY ESTIMATION PERFORMANCE")
    print("="*70)
    
    # Convert density score to float
    df['MADE_Density_Score_Float'] = df['MADE_Density_Score'].astype(float)
    
    print(f"\n  Density Score Statistics:")
    print(f"    Mean:   {df['MADE_Density_Score_Float'].mean():.4f}")
    print(f"    Median: {df['MADE_Density_Score_Float'].median():.4f}")
    print(f"    Std:    {df['MADE_Density_Score_Float'].std():.4f}")
    print(f"    Min:    {df['MADE_Density_Score_Float'].min():.4f}")
    print(f"    Max:    {df['MADE_Density_Score_Float'].max():.4f}")
    
    print(f"\n  Density Classification:")
    n_dense = df['MADE_Is_Dense'].sum()
    n_sparse = len(df) - n_dense
    print(f"    High density: {n_dense} ({n_dense/len(df)*100:.1f}%)")
    print(f"    Low density:  {n_sparse} ({n_sparse/len(df)*100:.1f}%)")
    
    # Density vs Noise
    print(f"\n  Density vs Noise:")
    dense_noise = ((df['MADE_Is_Dense'] == True) & (df['Is_Noise'] == True)).sum()
    dense_clean = ((df['MADE_Is_Dense'] == True) & (df['Is_Noise'] == False)).sum()
    sparse_noise = ((df['MADE_Is_Dense'] == False) & (df['Is_Noise'] == True)).sum()
    sparse_clean = ((df['MADE_Is_Dense'] == False) & (df['Is_Noise'] == False)).sum()
    
    print(f"    Dense + Noise:  {dense_noise} ({dense_noise/n_dense*100:.1f}% of dense)")
    print(f"    Dense + Clean:  {dense_clean} ({dense_clean/n_dense*100:.1f}% of dense)")
    print(f"    Sparse + Noise: {sparse_noise} ({sparse_noise/n_sparse*100:.1f}% of sparse)")
    print(f"    Sparse + Clean: {sparse_clean} ({sparse_clean/n_sparse*100:.1f}% of sparse)")


def analyze_knn_performance(df):
    """Analyze KNN semantic voting performance"""
    print("\n" + "="*70)
    print("KNN SEMANTIC VOTING PERFORMANCE")
    print("="*70)
    
    # KNN consistency
    df['KNN_Consistency_Float'] = df['KNN_Consistency'].astype(float)
    
    print(f"\n  Neighbor Consistency Statistics:")
    print(f"    Mean:   {df['KNN_Consistency_Float'].mean():.4f}")
    print(f"    Median: {df['KNN_Consistency_Float'].median():.4f}")
    print(f"    Std:    {df['KNN_Consistency_Float'].std():.4f}")
    print(f"    Min:    {df['KNN_Consistency_Float'].min():.4f}")
    print(f"    Max:    {df['KNN_Consistency_Float'].max():.4f}")
    
    # KNN vs Noisy label agreement
    knn_agree_noisy = (df['KNN_Neighbor_Label'] == df['Noisy_Label']).sum()
    print(f"\n  KNN Agreement with Noisy Labels:")
    print(f"    Agree:    {knn_agree_noisy} ({knn_agree_noisy/len(df)*100:.1f}%)")
    print(f"    Disagree: {len(df) - knn_agree_noisy} ({(len(df)-knn_agree_noisy)/len(df)*100:.1f}%)")
    
    # KNN vs CL agreement
    knn_agree_cl = (df['KNN_Neighbor_Label'] == df['CL_Pred_Label']).sum()
    print(f"\n  KNN Agreement with CL Predictions:")
    print(f"    Agree:    {knn_agree_cl} ({knn_agree_cl/len(df)*100:.1f}%)")
    print(f"    Disagree: {len(df) - knn_agree_cl} ({(len(df)-knn_agree_cl)/len(df)*100:.1f}%)")


def analyze_error_patterns(df):
    """Analyze error patterns in correction"""
    print("\n" + "="*70)
    print("ERROR PATTERN ANALYSIS")
    print("="*70)
    
    # Find incorrect corrections (excluding dropped samples)
    incorrect = df[df['Correction_Correct'] == 'Incorrect']
    
    if len(incorrect) == 0:
        print("\n  ✓ No incorrect corrections! Perfect performance.")
        return
    
    print(f"\n  Total incorrect corrections: {len(incorrect)}")
    
    # Error types
    print(f"\n  Error Types:")
    error_types = []
    for _, row in incorrect.iterrows():
        error_type = f"{row['True_Label']} → {row['Noisy_Label']} → {row['Corrected_Label']}"
        error_types.append(error_type)
    
    error_counts = Counter(error_types)
    for error_type, count in error_counts.most_common():
        print(f"    {error_type}: {count}")
    
    # Error distribution by action
    print(f"\n  Errors by Action:")
    for action in ['Keep', 'Flip', 'Reweight']:
        n_errors = len(incorrect[incorrect['Action'] == action])
        n_total_action = len(df[df['Action'] == action])
        if n_total_action > 0:
            error_rate = n_errors / n_total_action * 100
            print(f"    {action:12s}: {n_errors}/{n_total_action} ({error_rate:.2f}%)")
    
    # Common characteristics of errors
    print(f"\n  Characteristics of Incorrect Corrections:")
    print(f"    CL suspected noise: {incorrect['CL_Suspected_Noise'].sum()} ({incorrect['CL_Suspected_Noise'].sum()/len(incorrect)*100:.1f}%)")
    print(f"    MADE high density:  {incorrect['MADE_Is_Dense'].sum()} ({incorrect['MADE_Is_Dense'].sum()/len(incorrect)*100:.1f}%)")
    print(f"    Actually noise:     {incorrect['Is_Noise'].sum()} ({incorrect['Is_Noise'].sum()/len(incorrect)*100:.1f}%)")


def plot_component_comparison(df, save_dir="output/label_correction/analysis/figures"):
    """Plot comparison of different components"""
    print("\n" + "="*70)
    print("GENERATING COMPONENT COMPARISON PLOTS")
    print("="*70)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Component accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate accuracy for each component
    cl_accuracy = (df['CL_Pred_Label'] == df['True_Label']).mean() * 100
    knn_accuracy = (df['KNN_Neighbor_Label'] == df['True_Label']).mean() * 100
    
    # Final correction accuracy (excluding dropped)
    kept_samples = df[df['Action'] != 'Drop']
    final_accuracy = (kept_samples['Corrected_Label'] == kept_samples['True_Label']).mean() * 100
    
    components = ['CL\nPrediction', 'KNN\nNeighbor', 'Hybrid Court\n(Final)']
    accuracies = [cl_accuracy, knn_accuracy, final_accuracy]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    bars = ax.bar(components, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Component Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'component_accuracy_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")
    
    # 2. Action distribution by noise status
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Noisy samples
    ax = axes[0]
    noisy_df = df[df['Is_Noise'] == True]
    action_counts_noisy = noisy_df['Action'].value_counts()
    colors_action = ['#2ecc71', '#e74c3c', '#95a5a6', '#f39c12']
    
    wedges, texts, autotexts = ax.pie(action_counts_noisy, labels=action_counts_noisy.index,
                                        autopct='%1.1f%%', colors=colors_action,
                                        startangle=90, textprops={'fontsize': 10})
    ax.set_title('Actions on Noisy Samples', fontsize=12, fontweight='bold')
    
    # Clean samples
    ax = axes[1]
    clean_df = df[df['Is_Noise'] == False]
    action_counts_clean = clean_df['Action'].value_counts()
    
    wedges, texts, autotexts = ax.pie(action_counts_clean, labels=action_counts_clean.index,
                                        autopct='%1.1f%%', colors=colors_action,
                                        startangle=90, textprops={'fontsize': 10})
    ax.set_title('Actions on Clean Samples', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'action_distribution_by_noise.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def main():
    """Main analysis function"""
    print("\n" + "="*70)
    print("LABEL CORRECTION RESULTS ANALYSIS")
    print("="*70)
    print()
    
    # Load results
    try:
        df, data = load_results()
        print(f"✓ Successfully loaded analysis results")
        print(f"  CSV samples:  {len(df)}")
        print(f"  NPZ arrays:   {list(data.keys())}")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nPlease run label_correction_analysis.py first:")
        print("  python label_correction_analysis.py")
        return
    
    # Run analyses
    print_basic_statistics(df)
    analyze_by_action(df)
    analyze_cl_performance(df)
    analyze_made_performance(df)
    analyze_knn_performance(df)
    analyze_error_patterns(df)
    plot_component_comparison(df)
    
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE")
    print("="*70)
    print()


if __name__ == "__main__":
    main()

