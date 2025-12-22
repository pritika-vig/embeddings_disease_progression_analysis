#!/usr/bin/env python
"""
Plotting Script: Generalizability Analysis (Paper-Ready).

Key Story: Trajectory fidelity (τ) on reference diseases aligns with 
classification performance (F1) on held-out diseases.

Inputs:
- generalizability_results.csv
- full_manifold_evaluation_100.csv
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from matplotlib.lines import Line2D

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
import plotting.plot_config as pcfg

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
pcfg.set_icml_style()

OUTPUT_DIR = config.PLOTS_OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GENERALIZABILITY_FILE = config.RESULTS_DIR / "generalizability_results.csv"
TAU_FILE = config.FULL_RESULTS_OUTPUT_PATH


def load_data():
    """Load generalizability results and tau data."""
    gen_df = pd.read_csv(GENERALIZABILITY_FILE)
    tau_df = pd.read_csv(TAU_FILE)
    tau_df = tau_df[tau_df["embedding_type"] == "final_embedding"].copy()
    return gen_df, tau_df


# -----------------------------------------------------------------------------
# Figure 1: Same-Disease τ vs F1
# -----------------------------------------------------------------------------
def plot_same_disease_scatter(gen_df, tau_df):
    """
    Scatter plot of τ vs F1 for the SAME disease/model pair.
    Color = Progression, Shape = Model
    """
    df_5shot = gen_df[gen_df["n_shots"] == 5].copy()
    df_5shot = df_5shot[df_5shot["model"].isin(pcfg.MODEL_ORDER)]

    # Get same-disease tau
    tau_lookup = tau_df.set_index(["model", "progression"])["tau"].to_dict()
    df_5shot["same_disease_tau"] = df_5shot.apply(
        lambda r: tau_lookup.get((r["model"], r["target_progression"]), np.nan), axis=1
    )

    # Drop any rows with missing tau
    df_5shot = df_5shot.dropna(subset=["same_disease_tau"])

    fig, ax = plt.subplots(figsize=(7, 6))

    # Plot each point manually to control color and marker
    for _, row in df_5shot.iterrows():
        color = pcfg.PROGRESSION_COLORS.get(row["target_progression"], "#333333")
        marker = pcfg.MODEL_MARKERS.get(row["model"], "o")
        ax.scatter(
            row["same_disease_tau"],
            row["f1_score"],
            c=color,
            marker=marker,
            s=180 if marker not in ["P", "*"] else 220,
            alpha=0.85,
            edgecolor="white" if marker not in ["P", "*"] else color,
            linewidth=1.5,
            zorder=2,
        )

    # Compute correlation
    rho, p = spearmanr(df_5shot["same_disease_tau"], df_5shot["f1_score"])

    ax.set_xlabel(r"Trajectory Fidelity ($\tau$)", fontsize=12, fontweight="bold")
    ax.set_ylabel("5-Shot F1 Score", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Same-Disease: τ and F1 Are Correlated\nSpearman ρ = {rho:.2f} (p = {p:.3f})",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.3)
    sns.despine(ax=ax, trim=True)

    # --- Legends ---
    # Progression legend (colors)
    progression_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=pcfg.PROGRESSION_COLORS[p], markersize=10,
               markeredgecolor="white", label=pcfg.PROGRESSION_NAMES.get(p, p))
        for p in df_5shot["target_progression"].unique()
    ]

    # Model legend (shapes)
    model_handles = [
        Line2D([0], [0], marker=pcfg.MODEL_MARKERS.get(m, "o"), color="w",
               markerfacecolor="#555555" if pcfg.MODEL_MARKERS.get(m) not in ["P", "*"] else "w",
               markeredgecolor="#555555",
               markersize=10, label=pcfg.MODEL_LABELS.get(m, m))
        for m in pcfg.MODEL_ORDER
    ]

    # Place legends
    legend1 = ax.legend(
        handles=progression_handles,
        title="Disease",
        loc="upper left",
        bbox_to_anchor=(0.0, -0.12),
        ncol=2,
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )
    ax.add_artist(legend1)

    ax.legend(
        handles=model_handles,
        title="Model",
        loc="upper left",
        bbox_to_anchor=(0.55, -0.12),
        ncol=2,
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )

    plt.subplots_adjust(bottom=0.25)

    save_path = OUTPUT_DIR / "same_disease_tau_vs_f1.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {save_path}")

    return {"rho": rho, "p": p}


# -----------------------------------------------------------------------------
# Figure 2: Reference τ → Held-Out F1 (Cross-Task)
# -----------------------------------------------------------------------------
def plot_generalization_figure(df):
    """
    Paper-ready figure: scatter plot + correlation table.
    """
    df_5shot = df[df["n_shots"] == 5].copy()
    df_5shot = df_5shot[df_5shot["model"].isin(pcfg.MODEL_ORDER)]
    targets = sorted(df_5shot["target_progression"].unique())

    fig = plt.figure(figsize=(10, 5.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.6, 1], wspace=0.35)

    # --- Panel A: Scatter Plot ---
    ax_scatter = fig.add_subplot(gs[0])

    for _, row in df_5shot.iterrows():
        color = pcfg.PROGRESSION_COLORS.get(row["target_progression"], "#333333")
        marker = pcfg.MODEL_MARKERS.get(row["model"], "o")
        ax_scatter.scatter(
            row["ref_tau_avg"],
            row["f1_score"],
            c=color,
            marker=marker,
            s=180 if marker not in ["P", "*"] else 220,
            alpha=0.85,
            edgecolor="white" if marker not in ["P", "*"] else color,
            linewidth=1.5,
            zorder=2,
        )

    rho_overall, p_overall = spearmanr(df_5shot["ref_tau_avg"], df_5shot["f1_score"])

    ax_scatter.set_xlabel(r"Reference Manifold Quality (Avg $\tau$ on Training Set)", 
                          fontsize=11, fontweight="bold")
    ax_scatter.set_ylabel("Held-Out Accuracy (5-Shot F1)", 
                          fontsize=11, fontweight="bold")
    ax_scatter.set_title(
        f"Manifold Quality Orders Held-Out Performance\nPooled ρ = {rho_overall:.2f}",
        fontsize=12,
        fontweight="bold",
    )
    ax_scatter.grid(True, linestyle="--", alpha=0.3)
    sns.despine(ax=ax_scatter, trim=True)

    # Legends
    task_handles = [
        Line2D([0], [0], marker="o", color="w", 
               markerfacecolor=pcfg.PROGRESSION_COLORS.get(t, "#333333"), markersize=10, 
               markeredgecolor="white", label=pcfg.PROGRESSION_NAMES.get(t, t))
        for t in targets
    ]

    model_handles = [
        Line2D([0], [0], marker=pcfg.MODEL_MARKERS.get(m, "o"), color="w",
               markerfacecolor="#555555" if pcfg.MODEL_MARKERS.get(m) not in ["P", "*"] else "w",
               markeredgecolor="#555555",
               markersize=10, label=pcfg.MODEL_LABELS.get(m, m))
        for m in pcfg.MODEL_ORDER
    ]

    legend1 = ax_scatter.legend(
        handles=task_handles,
        title="Target Task",
        loc="upper left",
        bbox_to_anchor=(0.0, -0.15),
        ncol=2,
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )
    ax_scatter.add_artist(legend1)

    ax_scatter.legend(
        handles=model_handles,
        title="Model",
        loc="upper left",
        bbox_to_anchor=(0.55, -0.15),
        ncol=2,
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )

    # --- Panel B: Correlation Table ---
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis("off")

    stats_rows = []
    for target in targets:
        sub = df_5shot[df_5shot["target_progression"] == target]
        rho, p = spearmanr(sub["ref_tau_avg"], sub["f1_score"])
        display_name = pcfg.PROGRESSION_NAMES.get(target, target)
        stats_rows.append({
            "Target Task": display_name,
            "Spearman ρ": f"{rho:.2f}",
            "p-value": f"{p:.3f}" if p >= 0.001 else "<.001",
        })

    avg_rho = np.mean([float(r["Spearman ρ"]) for r in stats_rows])
    stats_rows.append({
        "Target Task": "AVERAGE",
        "Spearman ρ": f"{avg_rho:.2f}",
        "p-value": "—",
    })

    cols = ["Target Task", "Spearman ρ", "p-value"]
    cell_text = [[row[c] for c in cols] for row in stats_rows]

    table = ax_table.table(
        cellText=cell_text,
        colLabels=cols,
        loc="center",
        cellLoc="center",
        bbox=[0.05, 0.3, 0.9, 0.6],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)

    for col in range(len(cols)):
        table[(0, col)].set_text_props(fontweight="bold")
        table[(0, col)].set_facecolor("#e6e6e6")

    n_rows = len(stats_rows)
    for col in range(len(cols)):
        table[(n_rows, col)].set_text_props(fontweight="bold")
        table[(n_rows, col)].set_facecolor("#f5f5f5")

    ax_table.set_title(
        "Rank Correlation\n(Reference τ vs Held-Out F1)",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )

    plt.subplots_adjust(bottom=0.25)

    save_path = OUTPUT_DIR / "generalization_main_figure.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {save_path}")

    return {"pooled_rho": rho_overall, "avg_rho": avg_rho}


# -----------------------------------------------------------------------------
# Supplementary: Per-Task Panels
# -----------------------------------------------------------------------------
def plot_per_task_panels(df):
    """Supplementary figure showing each held-out task separately."""
    df_5shot = df[df["n_shots"] == 5].copy()
    df_5shot = df_5shot[df_5shot["model"].isin(pcfg.MODEL_ORDER)]
    targets = sorted(df_5shot["target_progression"].unique())

    fig, axes = plt.subplots(1, len(targets), figsize=(3.5 * len(targets), 4), sharey=True)

    for idx, target in enumerate(targets):
        ax = axes[idx]
        sub = df_5shot[df_5shot["target_progression"] == target]

        sns.scatterplot(
            data=sub,
            x="ref_tau_avg",
            y="f1_score",
            hue="model",
            palette=pcfg.MODEL_COLORS,
            hue_order=pcfg.MODEL_ORDER,
            s=180,
            alpha=0.85,
            edgecolor="white",
            linewidth=1,
            ax=ax,
            legend=(idx == len(targets) - 1),
        )

        rho, p = spearmanr(sub["ref_tau_avg"], sub["f1_score"])
        display_name = pcfg.PROGRESSION_NAMES.get(target, target)

        ax.set_title(f"Target: {display_name}\nρ = {rho:.2f} (p = {p:.3f})", 
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Reference τ", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Held-Out 5-Shot F1", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)

    if axes[-1].get_legend():
        axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=9)

    plt.suptitle("Per-Task: Reference τ vs Held-Out F1", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    save_path = OUTPUT_DIR / "generalization_per_task_supplement.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {save_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("PLOTTING: Generalizability Analysis")
    print("=" * 60)

    try:
        gen_df, tau_df = load_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Loaded {len(gen_df)} rows")

    # Figure 1: Same-disease τ vs F1
    same_disease_results = plot_same_disease_scatter(gen_df, tau_df)
    print(f"Same-disease: ρ = {same_disease_results['rho']:.2f}")

    # Figure 2: Cross-task generalization
    cross_task_results = plot_generalization_figure(gen_df)
    print(f"Cross-task: pooled ρ = {cross_task_results['pooled_rho']:.2f}, avg ρ = {cross_task_results['avg_rho']:.2f}")

    # Supplementary: Per-task panels
    plot_per_task_panels(gen_df)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()