"""
FusionNet — Non-IID Benchmark Plotter
======================================
Reads the metrics and partition reports from run_noniid_benchmark.py
and generates presentation-ready charts for the hackathon pitch.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CHARTS_DIR = RESULTS_DIR / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

# Set seaborn style for premium aesthetics
sns.set_theme(style="darkgrid", palette="pastel")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.facecolor'] = '#f8f9fa'


def load_data():
    metrics_path = RESULTS_DIR / "noniid_metrics.json"
    partition_path = RESULTS_DIR / "partition_report.json"
    
    if not metrics_path.exists() or not partition_path.exists():
        print("Data files not found. Run run_noniid_benchmark.py first.")
        return None, None
        
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    with open(partition_path, 'r') as f:
        partitions = json.load(f)
        
    return metrics, partitions


def plot_accuracy_convergence(metrics):
    rounds = [0] + [m["round"] for m in metrics]
    # Assuming baseline 0.4966 if not recorded in JSON as round 0
    # Let's derive baseline from the first round's text output or just set to ~0.50 for binary clf
    baseline = 0.4966 
    accs = [baseline * 100] + [m["accuracy_pct"] for m in metrics]
    
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, accs, marker='o', linewidth=3, markersize=8, color='#3b82f6', label='Global Model Accuracy')
    
    plt.axhline(y=baseline * 100, color='#ef4444', linestyle='--', alpha=0.7, label='Zero-shot Baseline')
    
    plt.title('Non-IID Federated Learning: Accuracy Convergence', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Federated Round', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(rounds)
    plt.ylim(40, max(accs) + 5)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    out_path = CHARTS_DIR / "01_accuracy_convergence.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path.name}")


def plot_client_loss(metrics):
    plt.figure(figsize=(10, 6))
    
    rounds = [m["round"] for m in metrics]
    
    # Extract client data
    client_losses = {}
    for m in metrics:
        for c in m["clients"]:
            tier = c["tier"]
            if tier not in client_losses:
                client_losses[tier] = []
            client_losses[tier].append(c["loss"])
            
    markers = ['o', 's', '^', 'D']
    colors = ['#f59e0b', '#10b981', '#8b5cf6', '#ec4899']
    
    for i, (tier, losses) in enumerate(client_losses.items()):
        plt.plot(rounds, losses, marker=markers[i%len(markers)], linewidth=2, 
                 markersize=7, color=colors[i%len(colors)], label=f'{tier} Node')
                 
    # Plot Global Avg
    avg_losses = [m["avg_loss"] for m in metrics]
    plt.plot(rounds, avg_losses, linestyle='--', color='#1f2937', linewidth=3, label='Global Average Loss')
    
    plt.title('Per-Tier Training Loss (Non-IID Data)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Federated Round', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.xticks(rounds)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    out_path = CHARTS_DIR / "02_client_loss.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path.name}")


def plot_data_heterogeneity(partitions):
    """Plot the Dirichlet Non-IID label distribution across tiers."""
    plt.figure(figsize=(12, 7))
    
    tiers = []
    pos_fractions = []
    neg_fractions = []
    
    # Process partition data
    for p in partitions:
        tier = f"Client {p['client_id']}\n({p['tier']})"
        tiers.append(tier)
        
        # In SST-2, 1 is positive, 0 is negative
        fracs = p["label_fractions"]
        pos = fracs.get("1", 0.0) * 100
        neg = fracs.get("0", 0.0) * 100
        
        pos_fractions.append(pos)
        neg_fractions.append(neg)
        
    x = np.arange(len(tiers))
    width = 0.6
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x, pos_fractions, width, label='Positive Sentiment', color='#34d399', alpha=0.9, edgecolor='black', linewidth=1)
    ax.bar(x, neg_fractions, width, bottom=pos_fractions, label='Negative Sentiment', color='#f87171', alpha=0.9, edgecolor='black', linewidth=1)
    
    ax.set_ylabel('Data Distribution (%)', fontsize=12)
    ax.set_title('Dirichlet Non-IID Data Skew by Hardware Tier', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(tiers, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.05), fontsize=11)
    
    # Add percentage labels
    for i in range(len(tiers)):
        if pos_fractions[i] > 5:
            ax.text(i, pos_fractions[i]/2, f'{pos_fractions[i]:.0f}%', ha='center', va='center', color='black', fontweight='bold')
        if neg_fractions[i] > 5:
            ax.text(i, pos_fractions[i] + neg_fractions[i]/2, f'{neg_fractions[i]:.0f}%', ha='center', va='center', color='black', fontweight='bold')

    plt.tight_layout()
    out_path = CHARTS_DIR / "03_data_heterogeneity.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"Saved {out_path.name}")


def plot_privacy_budget(metrics):
    plt.figure(figsize=(9, 5))
    
    rounds = [m["round"] for m in metrics]
    epsilons = [m["epsilon_avg"] for m in metrics]
    
    plt.fill_between(rounds, epsilons, alpha=0.2, color='#8b5cf6')
    plt.plot(rounds, epsilons, marker='D', color='#8b5cf6', linewidth=3, markersize=8)
    
    # Privacy budget cap
    plt.axhline(y=1.0, color='#ef4444', linestyle='--', linewidth=2, label='Strict Privacy Cap (ε = 1.0)')
    
    plt.title('Differential Privacy Budget (ε) Consumption', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Federated Round', fontsize=12)
    plt.ylabel('Cumulative Epsilon (ε)', fontsize=12)
    plt.xticks(rounds)
    plt.ylim(0, 1.2)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    out_path = CHARTS_DIR / "04_privacy_budget.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path.name}")


if __name__ == "__main__":
    print("Generating FusionNet benchmark charts...")
    metrics, partitions = load_data()
    if metrics and partitions:
        plot_accuracy_convergence(metrics)
        plot_client_loss(metrics)
        plot_data_heterogeneity(partitions)
        plot_privacy_budget(metrics)
        print(f"\nAll charts saved to: {CHARTS_DIR}")
