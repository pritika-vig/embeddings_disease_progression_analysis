#!/usr/bin/env python
"""
Plotting Script: Generalizability Analysis (Paper-Ready, ICML Style).

Produces:
1. Main Figure Panel A: Within-disease τ vs F1 correlations (per-disease + average)
2. Main Figure Panel B: Rank preservation panels (τ rank vs F1 rank, 1x4 grid)
3. Appendix Figure A: Reference τ vs held-out F1 scatter (pooled)
4. Appendix Figure B: τ vs F1 as reference metrics comparison (2x4 grid)

Inputs:
- generalizability_results.csv
- full_manifold_evaluation_100.csv
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr, linregress
from matplotlib.lines import Line2D

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
import plotting.plot_config as pcfg

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
pcfg.set_icml_style()

OUTPUT_DIR = config.PLOTS_OUTPUT_DIR / "generalization"
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
# Main Figure Panel A: Within-Disease τ vs F1 Correlations
# -----------------------------------------------------------------------------
def plot_within_disease_correlations(gen_df, tau_df):
    """
    Scatter plot of τ vs F1 for the SAME disease/model pair.
    Computes correlation WITHIN each disease progression separately,
    then reports individual correlations and the average.
    
    Color = Progression, Shape = Model
    Shows per-disease regression lines and correlations.
    """
    df_5shot = gen_df[gen_df["n_shots"] == 5].copy()
    df_5shot = df_5shot[df_5shot["model"].isin(pcfg.MODEL_ORDER)]

    # Get same-disease tau
    tau_lookup = tau_df.set_index(["model", "progression"])["tau"].to_dict()
    df_5shot["same_disease_tau"] = df_5shot.apply(
        lambda r: tau_lookup.get((r["model"], r["target_progression"]), np.nan), axis=1
    )
    df_5shot = df_5shot.dropna(subset=["same_disease_tau"])

    progressions = sorted(df_5shot["target_progression"].unique())
    
    fig, ax = plt.subplots(figsize=(7, 6))

    # Store per-disease correlations
    within_disease_corrs = {}
    within_disease_pearson = {}

    # Plot points and compute within-disease correlations
    for prog in progressions:
        prog_data = df_5shot[df_5shot["target_progression"] == prog]
        color = pcfg.PROGRESSION_COLORS.get(prog, "#333333")
        
        x = prog_data["same_disease_tau"].values
        y = prog_data["f1_score"].values
        
        # Compute within-disease correlations
        if len(x) >= 3:  # Need at least 3 points for meaningful correlation
            rho, p_spearman = spearmanr(x, y)
            r, p_pearson = pearsonr(x, y)
            within_disease_corrs[prog] = {"rho": rho, "p": p_spearman, "n": len(x)}
            within_disease_pearson[prog] = {"r": r, "p": p_pearson, "n": len(x)}
            
            # Fit regression line for this disease
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            x_line = np.linspace(x.min() - 0.01, x.max() + 0.01, 50)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color=color, linestyle='--', alpha=0.5, linewidth=1.5, zorder=1)
        else:
            within_disease_corrs[prog] = {"rho": np.nan, "p": np.nan, "n": len(x)}
            within_disease_pearson[prog] = {"r": np.nan, "p": np.nan, "n": len(x)}

        # Plot points
        for _, row in prog_data.iterrows():
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

    ax.set_xlabel(r"Trajectory Fidelity ($\tau$)", fontsize=11)
    ax.set_ylabel("5-Shot F1 Score", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.3)
    sns.despine(ax=ax, trim=True)

    # Compute average correlation (Spearman)
    valid_rhos = [v["rho"] for v in within_disease_corrs.values() if not np.isnan(v["rho"])]
    avg_rho = np.mean(valid_rhos) if valid_rhos else np.nan
    
    # Compute average Pearson r
    valid_rs = [v["r"] for v in within_disease_pearson.values() if not np.isnan(v["r"])]
    avg_r = np.mean(valid_rs) if valid_rs else np.nan

    # Build correlation text for display
    corr_lines = []
    for prog in progressions:
        prog_name = pcfg.PROGRESSION_NAMES.get(prog, prog)
        rho = within_disease_corrs[prog]["rho"]
        r = within_disease_pearson[prog]["r"]
        if not np.isnan(rho):
            corr_lines.append(f"{prog_name}: ρ={rho:.2f}, r={r:.2f}")
        else:
            corr_lines.append(f"{prog_name}: insufficient data")
    
    corr_lines.append(f"─────────────")
    corr_lines.append(f"Avg: ρ={avg_rho:.2f}, r={avg_r:.2f}")
    
    corr_text = "\n".join(corr_lines)
    
    ax.text(
        0.95, 0.05,
        corr_text,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=9,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
    )

    # Legends
    progression_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=pcfg.PROGRESSION_COLORS[p], markersize=10,
               markeredgecolor="white", label=pcfg.PROGRESSION_NAMES.get(p, p))
        for p in progressions
    ]

    model_handles = [
        Line2D([0], [0], marker=pcfg.MODEL_MARKERS.get(m, "o"), color="w",
               markerfacecolor="#555555" if pcfg.MODEL_MARKERS.get(m) not in ["P", "*"] else "w",
               markeredgecolor="#555555",
               markersize=10, label=pcfg.MODEL_LABELS.get(m, m))
        for m in pcfg.MODEL_ORDER
    ]

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

    save_path = OUTPUT_DIR / "main_panel_a_within_disease.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {save_path}")

    return {
        "within_disease_spearman": within_disease_corrs,
        "within_disease_pearson": within_disease_pearson,
        "avg_spearman_rho": avg_rho,
        "avg_pearson_r": avg_r,
    }


# -----------------------------------------------------------------------------
# Main Figure Panel B: Rank Preservation Panels (1x4 grid)
# -----------------------------------------------------------------------------
def plot_rank_preservation_panels(gen_df):
    """
    1x4 grid showing rank preservation: does τ ordering on reference tasks 
    predict F1 ranking on held-out task?
    
    X-axis: model rank by reference τ
    Y-axis: model rank by held-out F1
    """
    df_5shot = gen_df[gen_df["n_shots"] == 5].copy()
    df_5shot = df_5shot[df_5shot["model"].isin(pcfg.MODEL_ORDER)]
    targets = sorted(df_5shot["target_progression"].unique())

    fig, axes = plt.subplots(1, len(targets), figsize=(3.2 * len(targets), 3.5))
    
    rank_corrs = []

    for idx, target in enumerate(targets):
        ax = axes[idx]
        sub = df_5shot[df_5shot["target_progression"] == target].copy()
        
        # Compute ranks
        sub["tau_rank"] = sub["ref_tau_avg"].rank()
        sub["f1_rank"] = sub["f1_score"].rank()
        
        rho, p = spearmanr(sub["tau_rank"], sub["f1_rank"])
        rank_corrs.append(rho)

        # Plot each model with its color
        for _, row in sub.iterrows():
            model_color = pcfg.MODEL_COLORS.get(row["model"], "#333333")
            ax.scatter(
                row["tau_rank"],
                row["f1_rank"],
                c=model_color,
                s=120,
                alpha=0.9,
                edgecolor="white",
                linewidth=1,
                zorder=3,
            )
            # Add model label
            ax.annotate(
                pcfg.MODEL_LABELS.get(row["model"], row["model"]),
                (row["tau_rank"], row["f1_rank"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.8,
            )

        # Add diagonal reference line
        lims = [0.5, len(sub) + 0.5]
        ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1, zorder=1)
        
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        
        display_name = pcfg.PROGRESSION_NAMES.get(target, target)
        ax.set_xlabel(r"$\tau$ Rank (Reference)", fontsize=10)
        if idx == 0:
            ax.set_ylabel("F1 Rank (Held-Out)", fontsize=10)
        
        # Add target name and correlation as text annotation
        ax.text(
            0.05, 0.95, 
            f"Target: {display_name}\nRank ρ = {rho:.2f}",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=9,
        )
        
        ax.grid(True, linestyle="--", alpha=0.3)
        sns.despine(ax=ax)

    # Add overall average as figure text
    avg_rho = np.mean(rank_corrs)
    fig.text(
        0.5, 0.02, 
        f"Avg Rank ρ = {avg_rho:.2f}",
        ha="center", fontsize=11, fontweight="bold",
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    save_path = OUTPUT_DIR / "main_panel_b_rank_preservation.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {save_path}")

    return {"avg_rank_rho": avg_rho, "per_task_rho": dict(zip(targets, rank_corrs))}


# -----------------------------------------------------------------------------
# Appendix Figure A: Reference τ → Held-Out F1 Scatter (Cross-Task)
# -----------------------------------------------------------------------------
def plot_cross_task_scatter(gen_df):
    """
    Scatter plot: reference τ (x) vs held-out F1 (y), pooled across targets.
    Color = Target disease, Shape = Model
    """
    df_5shot = gen_df[gen_df["n_shots"] == 5].copy()
    df_5shot = df_5shot[df_5shot["model"].isin(pcfg.MODEL_ORDER)]
    targets = sorted(df_5shot["target_progression"].unique())

    fig, ax = plt.subplots(figsize=(7, 6))

    for _, row in df_5shot.iterrows():
        color = pcfg.PROGRESSION_COLORS.get(row["target_progression"], "#333333")
        marker = pcfg.MODEL_MARKERS.get(row["model"], "o")
        ax.scatter(
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

    # Compute per-target correlations and average
    per_target_rhos = []
    for target in targets:
        sub = df_5shot[df_5shot["target_progression"] == target]
        rho, _ = spearmanr(sub["ref_tau_avg"], sub["f1_score"])
        per_target_rhos.append(rho)

    avg_rho = np.mean(per_target_rhos)
    rho_overall, p_overall = spearmanr(df_5shot["ref_tau_avg"], df_5shot["f1_score"])

    ax.set_xlabel(r"Reference Manifold Quality (Avg $\tau$ on Training Set)", fontsize=11)
    ax.set_ylabel("Held-Out Accuracy (5-Shot F1)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.3)
    sns.despine(ax=ax, trim=True)

    # Add average correlation as text
    ax.text(
        0.95, 0.05,
        f"Avg ρ = {avg_rho:.2f}",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=12,
        fontweight="bold",
    )

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

    legend1 = ax.legend(
        handles=task_handles,
        title="Target Task",
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

    save_path = OUTPUT_DIR / "appendix_cross_task_scatter.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {save_path}")

    return {
        "pooled_rho": rho_overall,
        "avg_rho": avg_rho,
        "per_target_rhos": dict(zip(targets, per_target_rhos)),
    }


# -----------------------------------------------------------------------------
# Appendix Figure B: Reference τ vs Reference F1 Comparison (2x4 grid)
# -----------------------------------------------------------------------------
def plot_reference_metric_comparison(gen_df, tau_df):
    """
    2x4 grid comparing reference τ and reference F1 as predictors of held-out F1.
    Top row: Reference τ → Held-out F1
    Bottom row: Reference F1 → Held-out F1
    """
    df_5shot = gen_df[gen_df["n_shots"] == 5].copy()
    df_5shot = df_5shot[df_5shot["model"].isin(pcfg.MODEL_ORDER)]
    targets = sorted(df_5shot["target_progression"].unique())

    # Compute reference F1 (leave-one-out average)
    def get_ref_f1(row):
        other_targets = [t for t in targets if t != row["target_progression"]]
        ref_f1s = df_5shot[
            (df_5shot["model"] == row["model"]) &
            (df_5shot["target_progression"].isin(other_targets))
        ]["f1_score"]
        return ref_f1s.mean() if len(ref_f1s) > 0 else np.nan

    df_5shot["ref_f1_avg"] = df_5shot.apply(get_ref_f1, axis=1)

    fig, axes = plt.subplots(2, len(targets), figsize=(3.5 * len(targets), 6), sharey=True)

    tau_rhos = []
    f1_rhos = []

    for idx, target in enumerate(targets):
        sub = df_5shot[df_5shot["target_progression"] == target]
        display_name = pcfg.PROGRESSION_NAMES.get(target, target)

        # Top row: Reference τ
        ax_tau = axes[0, idx]
        for _, row in sub.iterrows():
            color = pcfg.MODEL_COLORS.get(row["model"], "#333333")
            ax_tau.scatter(
                row["ref_tau_avg"],
                row["f1_score"],
                c=color,
                s=120,
                alpha=0.85,
                edgecolor="white",
                linewidth=1,
            )

        rho_tau, _ = spearmanr(sub["ref_tau_avg"], sub["f1_score"])
        tau_rhos.append(rho_tau)

        # Text annotation instead of title
        ax_tau.text(
            0.05, 0.95,
            f"{display_name}\nτ: ρ = {rho_tau:.2f}",
            transform=ax_tau.transAxes,
            ha="left", va="top",
            fontsize=9,
        )
        ax_tau.set_xlabel(r"Reference $\tau$", fontsize=9)
        if idx == 0:
            ax_tau.set_ylabel("Held-Out F1", fontsize=10)
        ax_tau.grid(True, linestyle="--", alpha=0.3)
        sns.despine(ax=ax_tau)

        # Bottom row: Reference F1
        ax_f1 = axes[1, idx]
        for _, row in sub.iterrows():
            color = pcfg.MODEL_COLORS.get(row["model"], "#333333")
            ax_f1.scatter(
                row["ref_f1_avg"],
                row["f1_score"],
                c=color,
                s=120,
                alpha=0.85,
                edgecolor="white",
                linewidth=1,
            )

        rho_f1, _ = spearmanr(sub["ref_f1_avg"], sub["f1_score"])
        f1_rhos.append(rho_f1)

        # Text annotation instead of title
        ax_f1.text(
            0.05, 0.95,
            f"F1: ρ = {rho_f1:.2f}",
            transform=ax_f1.transAxes,
            ha="left", va="top",
            fontsize=9,
        )
        ax_f1.set_xlabel("Reference F1", fontsize=9)
        if idx == 0:
            ax_f1.set_ylabel("Held-Out F1", fontsize=10)
        ax_f1.grid(True, linestyle="--", alpha=0.3)
        sns.despine(ax=ax_f1)

    # Add row labels
    fig.text(0.02, 0.72, r"Reference" + "\n" + r"$\tau$", ha="center", va="center",
             fontsize=11, fontweight="bold", rotation=90)
    fig.text(0.02, 0.28, "Reference\nF1", ha="center", va="center",
             fontsize=11, fontweight="bold", rotation=90)

    avg_tau_rho = np.mean(tau_rhos)
    avg_f1_rho = np.mean(f1_rhos)

    plt.tight_layout()
    plt.subplots_adjust(left=0.08)

    save_path = OUTPUT_DIR / "appendix_reference_metric_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {save_path}")

    return {
        "avg_tau_rho": avg_tau_rho,
        "avg_f1_rho": avg_f1_rho,
        "per_target_tau_rhos": dict(zip(targets, tau_rhos)),
        "per_target_f1_rhos": dict(zip(targets, f1_rhos)),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("PLOTTING: Generalizability Analysis (ICML Style)")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

    try:
        gen_df, tau_df = load_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Loaded {len(gen_df)} rows")

    # Main Figure Panel A: Within-disease τ vs F1 correlations
    within_disease_results = plot_within_disease_correlations(gen_df, tau_df)

    # Main Figure Panel B: Rank preservation panels
    rank_results = plot_rank_preservation_panels(gen_df)

    # Appendix Figure A: Cross-task scatter
    cross_task_results = plot_cross_task_scatter(gen_df)

    # Appendix Figure B: Reference metric comparison
    comparison_results = plot_reference_metric_comparison(gen_df, tau_df)

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS FOR PAPER:")
    print("=" * 60)
    print(f"\n  WITHIN-DISEASE τ vs F1 CORRELATIONS:")
    print(f"  " + "-" * 50)
    for prog, stats in within_disease_results["within_disease_spearman"].items():
        prog_name = pcfg.PROGRESSION_NAMES.get(prog, prog)
        pearson_r = within_disease_results["within_disease_pearson"][prog]["r"]
        print(f"    {prog_name:20s}: ρ = {stats['rho']:+.2f}, r = {pearson_r:+.2f} (n={stats['n']})")
    print(f"  " + "-" * 50)
    print(f"    {'AVERAGE':20s}: ρ = {within_disease_results['avg_spearman_rho']:+.2f}, r = {within_disease_results['avg_pearson_r']:+.2f}")
    print()
    print(f"  Rank preservation (avg):  ρ = {rank_results['avg_rank_rho']:.2f}")
    print(f"  Per-target rank ρ:        {rank_results['per_task_rho']}")
    print(f"  Cross-task (pooled):      ρ = {cross_task_results['pooled_rho']:.2f}")
    print(f"  Cross-task (avg):         ρ = {cross_task_results['avg_rho']:.2f}")
    print(f"  Reference τ avg:          ρ = {comparison_results['avg_tau_rho']:.2f}")
    print(f"  Reference F1 avg:         ρ = {comparison_results['avg_f1_rho']:.2f}")
    print("=" * 60)

    print("\n✅ Done!")
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("  MAIN FIGURES:")
    print("    - main_panel_a_within_disease.png")
    print("    - main_panel_b_rank_preservation.png")
    print("  APPENDIX FIGURES:")
    print("    - appendix_cross_task_scatter.png")
    print("    - appendix_reference_metric_comparison.png")


if __name__ == "__main__":
    main()
# #!/usr/bin/env python
# """
# Plotting Script: Generalizability Analysis (Paper-Ready, ICML Style).

# Produces:
# 1. Main Figure Panel A: Same-disease τ vs F1 (with linear regression)
# 2. Main Figure Panel B: Rank preservation panels (τ rank vs F1 rank, 1x4 grid)
# 3. Appendix Figure A: Reference τ vs held-out F1 scatter (pooled)
# 4. Appendix Figure B: τ vs F1 as reference metrics comparison (2x4 grid)

# Inputs:
# - generalizability_results.csv
# - full_manifold_evaluation_100.csv
# """

# import sys
# from pathlib import Path
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import numpy as np
# from scipy.stats import spearmanr, pearsonr, linregress
# from matplotlib.lines import Line2D

# sys.path.append(str(Path(__file__).resolve().parent.parent))

# import config
# import plotting.plot_config as pcfg

# # -----------------------------------------------------------------------------
# # Configuration
# # -----------------------------------------------------------------------------
# pcfg.set_icml_style()

# OUTPUT_DIR = config.PLOTS_OUTPUT_DIR / "generalization"
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# GENERALIZABILITY_FILE = config.RESULTS_DIR / "generalizability_results.csv"
# TAU_FILE = config.FULL_RESULTS_OUTPUT_PATH


# def load_data():
#     """Load generalizability results and tau data."""
#     gen_df = pd.read_csv(GENERALIZABILITY_FILE)
#     tau_df = pd.read_csv(TAU_FILE)
#     tau_df = tau_df[tau_df["embedding_type"] == "final_embedding"].copy()
#     return gen_df, tau_df


# # -----------------------------------------------------------------------------
# # Main Figure Panel A: Same-Disease τ vs F1 (Linear Regression)
# # -----------------------------------------------------------------------------
# def plot_same_disease_scatter(gen_df, tau_df):
#     """
#     Scatter plot of τ vs F1 for the SAME disease/model pair.
#     Color = Progression, Shape = Model
#     Shows linear regression fit (F1 = m*τ + b)
#     """
#     df_5shot = gen_df[gen_df["n_shots"] == 5].copy()
#     df_5shot = df_5shot[df_5shot["model"].isin(pcfg.MODEL_ORDER)]

#     # Get same-disease tau
#     tau_lookup = tau_df.set_index(["model", "progression"])["tau"].to_dict()
#     df_5shot["same_disease_tau"] = df_5shot.apply(
#         lambda r: tau_lookup.get((r["model"], r["target_progression"]), np.nan), axis=1
#     )
#     df_5shot = df_5shot.dropna(subset=["same_disease_tau"])

#     fig, ax = plt.subplots(figsize=(7, 6))

#     for _, row in df_5shot.iterrows():
#         color = pcfg.PROGRESSION_COLORS.get(row["target_progression"], "#333333")
#         marker = pcfg.MODEL_MARKERS.get(row["model"], "o")
#         ax.scatter(
#             row["same_disease_tau"],
#             row["f1_score"],
#             c=color,
#             marker=marker,
#             s=180 if marker not in ["P", "*"] else 220,
#             alpha=0.85,
#             edgecolor="white" if marker not in ["P", "*"] else color,
#             linewidth=1.5,
#             zorder=2,
#         )

#     # Linear regression
#     x = df_5shot["same_disease_tau"].values
#     y = df_5shot["f1_score"].values
#     slope, intercept, r_value, p_value, std_err = linregress(x, y)
#     r_squared = r_value ** 2
    
#     # Plot regression line
#     x_line = np.linspace(x.min() - 0.02, x.max() + 0.02, 100)
#     y_line = slope * x_line + intercept
#     ax.plot(x_line, y_line, 'k--', alpha=0.5, linewidth=1.5, zorder=1)

#     ax.set_xlabel(r"Trajectory Fidelity ($\tau$)", fontsize=11)
#     ax.set_ylabel("5-Shot F1 Score", fontsize=11)
#     ax.grid(True, linestyle="--", alpha=0.3)
#     sns.despine(ax=ax, trim=True)

#     # Add regression equation and R²
#     ax.text(
#         0.95, 0.05,
#         f"F1 = {slope:.2f}τ + {intercept:.2f}\n$R^2$ = {r_squared:.2f}",
#         transform=ax.transAxes,
#         ha="right", va="bottom",
#         fontsize=11,
#         fontweight="bold",
#     )

#     # Legends
#     progression_handles = [
#         Line2D([0], [0], marker="o", color="w",
#                markerfacecolor=pcfg.PROGRESSION_COLORS[p], markersize=10,
#                markeredgecolor="white", label=pcfg.PROGRESSION_NAMES.get(p, p))
#         for p in df_5shot["target_progression"].unique()
#     ]

#     model_handles = [
#         Line2D([0], [0], marker=pcfg.MODEL_MARKERS.get(m, "o"), color="w",
#                markerfacecolor="#555555" if pcfg.MODEL_MARKERS.get(m) not in ["P", "*"] else "w",
#                markeredgecolor="#555555",
#                markersize=10, label=pcfg.MODEL_LABELS.get(m, m))
#         for m in pcfg.MODEL_ORDER
#     ]

#     legend1 = ax.legend(
#         handles=progression_handles,
#         title="Disease",
#         loc="upper left",
#         bbox_to_anchor=(0.0, -0.12),
#         ncol=2,
#         frameon=False,
#         fontsize=9,
#         title_fontsize=10,
#     )
#     ax.add_artist(legend1)

#     ax.legend(
#         handles=model_handles,
#         title="Model",
#         loc="upper left",
#         bbox_to_anchor=(0.55, -0.12),
#         ncol=2,
#         frameon=False,
#         fontsize=9,
#         title_fontsize=10,
#     )

#     plt.subplots_adjust(bottom=0.25)

#     save_path = OUTPUT_DIR / "main_panel_a_same_disease.png"
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.close()
#     print(f"✅ Saved: {save_path}")

#     return {"slope": slope, "intercept": intercept, "r_squared": r_squared, "p": p_value}


# # -----------------------------------------------------------------------------
# # Main Figure Panel B: Rank Preservation Panels (1x4 grid)
# # -----------------------------------------------------------------------------
# def plot_rank_preservation_panels(gen_df):
#     """
#     1x4 grid showing rank preservation: does τ ordering on reference tasks 
#     predict F1 ranking on held-out task?
    
#     X-axis: model rank by reference τ
#     Y-axis: model rank by held-out F1
#     """
#     df_5shot = gen_df[gen_df["n_shots"] == 5].copy()
#     df_5shot = df_5shot[df_5shot["model"].isin(pcfg.MODEL_ORDER)]
#     targets = sorted(df_5shot["target_progression"].unique())

#     fig, axes = plt.subplots(1, len(targets), figsize=(3.2 * len(targets), 3.5))
    
#     rank_corrs = []

#     for idx, target in enumerate(targets):
#         ax = axes[idx]
#         sub = df_5shot[df_5shot["target_progression"] == target].copy()
        
#         # Compute ranks
#         sub["tau_rank"] = sub["ref_tau_avg"].rank()
#         sub["f1_rank"] = sub["f1_score"].rank()
        
#         rho, p = spearmanr(sub["tau_rank"], sub["f1_rank"])
#         rank_corrs.append(rho)

#         # Plot each model with its color
#         for _, row in sub.iterrows():
#             model_color = pcfg.MODEL_COLORS.get(row["model"], "#333333")
#             ax.scatter(
#                 row["tau_rank"],
#                 row["f1_rank"],
#                 c=model_color,
#                 s=120,
#                 alpha=0.9,
#                 edgecolor="white",
#                 linewidth=1,
#                 zorder=3,
#             )
#             # Add model label
#             ax.annotate(
#                 pcfg.MODEL_LABELS.get(row["model"], row["model"]),
#                 (row["tau_rank"], row["f1_rank"]),
#                 xytext=(5, 5),
#                 textcoords="offset points",
#                 fontsize=8,
#                 alpha=0.8,
#             )

#         # Add diagonal reference line
#         lims = [0.5, len(sub) + 0.5]
#         ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1, zorder=1)
        
#         ax.set_xlim(lims)
#         ax.set_ylim(lims)
#         ax.set_aspect("equal")
        
#         display_name = pcfg.PROGRESSION_NAMES.get(target, target)
#         ax.set_xlabel(r"$\tau$ Rank (Reference)", fontsize=10)
#         if idx == 0:
#             ax.set_ylabel("F1 Rank (Held-Out)", fontsize=10)
        
#         # Add target name and correlation as text annotation
#         ax.text(
#             0.05, 0.95, 
#             f"Target: {display_name}\nRank ρ = {rho:.2f}",
#             transform=ax.transAxes,
#             ha="left", va="top",
#             fontsize=9,
#         )
        
#         ax.grid(True, linestyle="--", alpha=0.3)
#         sns.despine(ax=ax)

#     # Add overall average as figure text
#     avg_rho = np.mean(rank_corrs)
#     fig.text(
#         0.5, 0.02, 
#         f"Avg Rank ρ = {avg_rho:.2f}",
#         ha="center", fontsize=11, fontweight="bold",
#     )

#     plt.tight_layout()
#     plt.subplots_adjust(bottom=0.15)

#     save_path = OUTPUT_DIR / "main_panel_b_rank_preservation.png"
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.close()
#     print(f"✅ Saved: {save_path}")

#     return {"avg_rank_rho": avg_rho, "per_task_rho": dict(zip(targets, rank_corrs))}


# # -----------------------------------------------------------------------------
# # Appendix Figure A: Reference τ → Held-Out F1 Scatter (Cross-Task)
# # -----------------------------------------------------------------------------
# def plot_cross_task_scatter(gen_df):
#     """
#     Scatter plot: reference τ (x) vs held-out F1 (y), pooled across targets.
#     Color = Target disease, Shape = Model
#     """
#     df_5shot = gen_df[gen_df["n_shots"] == 5].copy()
#     df_5shot = df_5shot[df_5shot["model"].isin(pcfg.MODEL_ORDER)]
#     targets = sorted(df_5shot["target_progression"].unique())

#     fig, ax = plt.subplots(figsize=(7, 6))

#     for _, row in df_5shot.iterrows():
#         color = pcfg.PROGRESSION_COLORS.get(row["target_progression"], "#333333")
#         marker = pcfg.MODEL_MARKERS.get(row["model"], "o")
#         ax.scatter(
#             row["ref_tau_avg"],
#             row["f1_score"],
#             c=color,
#             marker=marker,
#             s=180 if marker not in ["P", "*"] else 220,
#             alpha=0.85,
#             edgecolor="white" if marker not in ["P", "*"] else color,
#             linewidth=1.5,
#             zorder=2,
#         )

#     # Compute per-target correlations and average
#     per_target_rhos = []
#     for target in targets:
#         sub = df_5shot[df_5shot["target_progression"] == target]
#         rho, _ = spearmanr(sub["ref_tau_avg"], sub["f1_score"])
#         per_target_rhos.append(rho)

#     avg_rho = np.mean(per_target_rhos)
#     rho_overall, p_overall = spearmanr(df_5shot["ref_tau_avg"], df_5shot["f1_score"])

#     ax.set_xlabel(r"Reference Manifold Quality (Avg $\tau$ on Training Set)", fontsize=11)
#     ax.set_ylabel("Held-Out Accuracy (5-Shot F1)", fontsize=11)
#     ax.grid(True, linestyle="--", alpha=0.3)
#     sns.despine(ax=ax, trim=True)

#     # Add average correlation as text
#     ax.text(
#         0.95, 0.05,
#         f"Avg ρ = {avg_rho:.2f}",
#         transform=ax.transAxes,
#         ha="right", va="bottom",
#         fontsize=12,
#         fontweight="bold",
#     )

#     # Legends
#     task_handles = [
#         Line2D([0], [0], marker="o", color="w",
#                markerfacecolor=pcfg.PROGRESSION_COLORS.get(t, "#333333"), markersize=10,
#                markeredgecolor="white", label=pcfg.PROGRESSION_NAMES.get(t, t))
#         for t in targets
#     ]

#     model_handles = [
#         Line2D([0], [0], marker=pcfg.MODEL_MARKERS.get(m, "o"), color="w",
#                markerfacecolor="#555555" if pcfg.MODEL_MARKERS.get(m) not in ["P", "*"] else "w",
#                markeredgecolor="#555555",
#                markersize=10, label=pcfg.MODEL_LABELS.get(m, m))
#         for m in pcfg.MODEL_ORDER
#     ]

#     legend1 = ax.legend(
#         handles=task_handles,
#         title="Target Task",
#         loc="upper left",
#         bbox_to_anchor=(0.0, -0.12),
#         ncol=2,
#         frameon=False,
#         fontsize=9,
#         title_fontsize=10,
#     )
#     ax.add_artist(legend1)

#     ax.legend(
#         handles=model_handles,
#         title="Model",
#         loc="upper left",
#         bbox_to_anchor=(0.55, -0.12),
#         ncol=2,
#         frameon=False,
#         fontsize=9,
#         title_fontsize=10,
#     )

#     plt.subplots_adjust(bottom=0.25)

#     save_path = OUTPUT_DIR / "appendix_cross_task_scatter.png"
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.close()
#     print(f"✅ Saved: {save_path}")

#     return {
#         "pooled_rho": rho_overall,
#         "avg_rho": avg_rho,
#         "per_target_rhos": dict(zip(targets, per_target_rhos)),
#     }


# # -----------------------------------------------------------------------------
# # Appendix Figure B: Reference τ vs Reference F1 Comparison (2x4 grid)
# # -----------------------------------------------------------------------------
# def plot_reference_metric_comparison(gen_df, tau_df):
#     """
#     2x4 grid comparing reference τ and reference F1 as predictors of held-out F1.
#     Top row: Reference τ → Held-out F1
#     Bottom row: Reference F1 → Held-out F1
#     """
#     df_5shot = gen_df[gen_df["n_shots"] == 5].copy()
#     df_5shot = df_5shot[df_5shot["model"].isin(pcfg.MODEL_ORDER)]
#     targets = sorted(df_5shot["target_progression"].unique())

#     # Compute reference F1 (leave-one-out average)
#     def get_ref_f1(row):
#         other_targets = [t for t in targets if t != row["target_progression"]]
#         ref_f1s = df_5shot[
#             (df_5shot["model"] == row["model"]) &
#             (df_5shot["target_progression"].isin(other_targets))
#         ]["f1_score"]
#         return ref_f1s.mean() if len(ref_f1s) > 0 else np.nan

#     df_5shot["ref_f1_avg"] = df_5shot.apply(get_ref_f1, axis=1)

#     fig, axes = plt.subplots(2, len(targets), figsize=(3.5 * len(targets), 6), sharey=True)

#     tau_rhos = []
#     f1_rhos = []

#     for idx, target in enumerate(targets):
#         sub = df_5shot[df_5shot["target_progression"] == target]
#         display_name = pcfg.PROGRESSION_NAMES.get(target, target)

#         # Top row: Reference τ
#         ax_tau = axes[0, idx]
#         for _, row in sub.iterrows():
#             color = pcfg.MODEL_COLORS.get(row["model"], "#333333")
#             ax_tau.scatter(
#                 row["ref_tau_avg"],
#                 row["f1_score"],
#                 c=color,
#                 s=120,
#                 alpha=0.85,
#                 edgecolor="white",
#                 linewidth=1,
#             )

#         rho_tau, _ = spearmanr(sub["ref_tau_avg"], sub["f1_score"])
#         tau_rhos.append(rho_tau)

#         # Text annotation instead of title
#         ax_tau.text(
#             0.05, 0.95,
#             f"{display_name}\nτ: ρ = {rho_tau:.2f}",
#             transform=ax_tau.transAxes,
#             ha="left", va="top",
#             fontsize=9,
#         )
#         ax_tau.set_xlabel(r"Reference $\tau$", fontsize=9)
#         if idx == 0:
#             ax_tau.set_ylabel("Held-Out F1", fontsize=10)
#         ax_tau.grid(True, linestyle="--", alpha=0.3)
#         sns.despine(ax=ax_tau)

#         # Bottom row: Reference F1
#         ax_f1 = axes[1, idx]
#         for _, row in sub.iterrows():
#             color = pcfg.MODEL_COLORS.get(row["model"], "#333333")
#             ax_f1.scatter(
#                 row["ref_f1_avg"],
#                 row["f1_score"],
#                 c=color,
#                 s=120,
#                 alpha=0.85,
#                 edgecolor="white",
#                 linewidth=1,
#             )

#         rho_f1, _ = spearmanr(sub["ref_f1_avg"], sub["f1_score"])
#         f1_rhos.append(rho_f1)

#         # Text annotation instead of title
#         ax_f1.text(
#             0.05, 0.95,
#             f"F1: ρ = {rho_f1:.2f}",
#             transform=ax_f1.transAxes,
#             ha="left", va="top",
#             fontsize=9,
#         )
#         ax_f1.set_xlabel("Reference F1", fontsize=9)
#         if idx == 0:
#             ax_f1.set_ylabel("Held-Out F1", fontsize=10)
#         ax_f1.grid(True, linestyle="--", alpha=0.3)
#         sns.despine(ax=ax_f1)

#     # Add row labels
#     fig.text(0.02, 0.72, r"Reference" + "\n" + r"$\tau$", ha="center", va="center",
#              fontsize=11, fontweight="bold", rotation=90)
#     fig.text(0.02, 0.28, "Reference\nF1", ha="center", va="center",
#              fontsize=11, fontweight="bold", rotation=90)

#     avg_tau_rho = np.mean(tau_rhos)
#     avg_f1_rho = np.mean(f1_rhos)

#     plt.tight_layout()
#     plt.subplots_adjust(left=0.08)

#     save_path = OUTPUT_DIR / "appendix_reference_metric_comparison.png"
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.close()
#     print(f"✅ Saved: {save_path}")

#     return {
#         "avg_tau_rho": avg_tau_rho,
#         "avg_f1_rho": avg_f1_rho,
#         "per_target_tau_rhos": dict(zip(targets, tau_rhos)),
#         "per_target_f1_rhos": dict(zip(targets, f1_rhos)),
#     }


# # -----------------------------------------------------------------------------
# # Main
# # -----------------------------------------------------------------------------
# def main():
#     print("=" * 60)
#     print("PLOTTING: Generalizability Analysis (ICML Style)")
#     print(f"Output directory: {OUTPUT_DIR}")
#     print("=" * 60)

#     try:
#         gen_df, tau_df = load_data()
#     except FileNotFoundError as e:
#         print(f"Error: {e}")
#         sys.exit(1)

#     print(f"Loaded {len(gen_df)} rows")

#     # Main Figure Panel A: Same-disease τ vs F1 (linear regression)
#     same_disease_results = plot_same_disease_scatter(gen_df, tau_df)

#     # Main Figure Panel B: Rank preservation panels
#     rank_results = plot_rank_preservation_panels(gen_df)

#     # Appendix Figure A: Cross-task scatter
#     cross_task_results = plot_cross_task_scatter(gen_df)

#     # Appendix Figure B: Reference metric comparison
#     comparison_results = plot_reference_metric_comparison(gen_df, tau_df)

#     print("\n" + "=" * 60)
#     print("SUMMARY STATISTICS FOR PAPER:")
#     print("=" * 60)
#     print(f"  Same-disease (linear reg):")
#     print(f"    F1 = {same_disease_results['slope']:.2f}τ + {same_disease_results['intercept']:.2f}")
#     print(f"    R² = {same_disease_results['r_squared']:.2f} (p = {same_disease_results['p']:.3f})")
#     print(f"  Rank preservation (avg):  ρ = {rank_results['avg_rank_rho']:.2f}")
#     print(f"  Per-target rank ρ:        {rank_results['per_task_rho']}")
#     print(f"  Cross-task (pooled):      ρ = {cross_task_results['pooled_rho']:.2f}")
#     print(f"  Cross-task (avg):         ρ = {cross_task_results['avg_rho']:.2f}")
#     print(f"  Reference τ avg:          ρ = {comparison_results['avg_tau_rho']:.2f}")
#     print(f"  Reference F1 avg:         ρ = {comparison_results['avg_f1_rho']:.2f}")
#     print("=" * 60)

#     print("\n✅ Done!")
#     print(f"\nOutputs saved to: {OUTPUT_DIR}")
#     print("  MAIN FIGURES:")
#     print("    - main_panel_a_same_disease.png")
#     print("    - main_panel_b_rank_preservation.png")
#     print("  APPENDIX FIGURES:")
#     print("    - appendix_cross_task_scatter.png")
#     print("    - appendix_reference_metric_comparison.png")


# if __name__ == "__main__":
#     main()
# #!/usr/bin/env python
# """
# Plotting Script: Generalizability Analysis (Paper-Ready).

# Key Story: Trajectory fidelity (τ) on reference diseases aligns with 
# classification performance (F1) on held-out diseases.

# Inputs:
# - generalizability_results.csv
# - full_manifold_evaluation_100.csv
# """

# import sys
# from pathlib import Path
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import seaborn as sns
# import pandas as pd
# import numpy as np
# from scipy.stats import spearmanr
# from matplotlib.lines import Line2D

# sys.path.append(str(Path(__file__).resolve().parent.parent))

# import config
# import plotting.plot_config as pcfg

# # -----------------------------------------------------------------------------
# # Configuration
# # -----------------------------------------------------------------------------
# pcfg.set_icml_style()

# OUTPUT_DIR = config.PLOTS_OUTPUT_DIR
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# GENERALIZABILITY_FILE = config.RESULTS_DIR / "generalizability_results.csv"
# TAU_FILE = config.FULL_RESULTS_OUTPUT_PATH


# def load_data():
#     """Load generalizability results and tau data."""
#     gen_df = pd.read_csv(GENERALIZABILITY_FILE)
#     tau_df = pd.read_csv(TAU_FILE)
#     tau_df = tau_df[tau_df["embedding_type"] == "final_embedding"].copy()
#     return gen_df, tau_df


# # -----------------------------------------------------------------------------
# # Figure 1: Same-Disease τ vs F1
# # -----------------------------------------------------------------------------
# def plot_same_disease_scatter(gen_df, tau_df):
#     """
#     Scatter plot of τ vs F1 for the SAME disease/model pair.
#     Color = Progression, Shape = Model
#     """
#     df_5shot = gen_df[gen_df["n_shots"] == 5].copy()
#     df_5shot = df_5shot[df_5shot["model"].isin(pcfg.MODEL_ORDER)]

#     # Get same-disease tau
#     tau_lookup = tau_df.set_index(["model", "progression"])["tau"].to_dict()
#     df_5shot["same_disease_tau"] = df_5shot.apply(
#         lambda r: tau_lookup.get((r["model"], r["target_progression"]), np.nan), axis=1
#     )

#     # Drop any rows with missing tau
#     df_5shot = df_5shot.dropna(subset=["same_disease_tau"])

#     fig, ax = plt.subplots(figsize=(7, 6))

#     # Plot each point manually to control color and marker
#     for _, row in df_5shot.iterrows():
#         color = pcfg.PROGRESSION_COLORS.get(row["target_progression"], "#333333")
#         marker = pcfg.MODEL_MARKERS.get(row["model"], "o")
#         ax.scatter(
#             row["same_disease_tau"],
#             row["f1_score"],
#             c=color,
#             marker=marker,
#             s=180 if marker not in ["P", "*"] else 220,
#             alpha=0.85,
#             edgecolor="white" if marker not in ["P", "*"] else color,
#             linewidth=1.5,
#             zorder=2,
#         )

#     # Compute correlation
#     rho, p = spearmanr(df_5shot["same_disease_tau"], df_5shot["f1_score"])

#     ax.set_xlabel(r"Trajectory Fidelity ($\tau$)", fontsize=12, fontweight="bold")
#     ax.set_ylabel("5-Shot F1 Score", fontsize=12, fontweight="bold")
#     ax.set_title(
#         f"Same-Disease: τ and F1 Are Correlated\nSpearman ρ = {rho:.2f} (p = {p:.3f})",
#         fontsize=13,
#         fontweight="bold",
#     )
#     ax.grid(True, linestyle="--", alpha=0.3)
#     sns.despine(ax=ax, trim=True)

#     # --- Legends ---
#     # Progression legend (colors)
#     progression_handles = [
#         Line2D([0], [0], marker="o", color="w",
#                markerfacecolor=pcfg.PROGRESSION_COLORS[p], markersize=10,
#                markeredgecolor="white", label=pcfg.PROGRESSION_NAMES.get(p, p))
#         for p in df_5shot["target_progression"].unique()
#     ]

#     # Model legend (shapes)
#     model_handles = [
#         Line2D([0], [0], marker=pcfg.MODEL_MARKERS.get(m, "o"), color="w",
#                markerfacecolor="#555555" if pcfg.MODEL_MARKERS.get(m) not in ["P", "*"] else "w",
#                markeredgecolor="#555555",
#                markersize=10, label=pcfg.MODEL_LABELS.get(m, m))
#         for m in pcfg.MODEL_ORDER
#     ]

#     # Place legends
#     legend1 = ax.legend(
#         handles=progression_handles,
#         title="Disease",
#         loc="upper left",
#         bbox_to_anchor=(0.0, -0.12),
#         ncol=2,
#         frameon=False,
#         fontsize=9,
#         title_fontsize=10,
#     )
#     ax.add_artist(legend1)

#     ax.legend(
#         handles=model_handles,
#         title="Model",
#         loc="upper left",
#         bbox_to_anchor=(0.55, -0.12),
#         ncol=2,
#         frameon=False,
#         fontsize=9,
#         title_fontsize=10,
#     )

#     plt.subplots_adjust(bottom=0.25)

#     save_path = OUTPUT_DIR / "same_disease_tau_vs_f1.png"
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.close()
#     print(f"✅ Saved: {save_path}")

#     return {"rho": rho, "p": p}


# # -----------------------------------------------------------------------------
# # Figure 2: Reference τ → Held-Out F1 (Cross-Task)
# # -----------------------------------------------------------------------------
# def plot_generalization_figure(df):
#     """
#     Paper-ready figure: scatter plot + correlation table.
#     """
#     df_5shot = df[df["n_shots"] == 5].copy()
#     df_5shot = df_5shot[df_5shot["model"].isin(pcfg.MODEL_ORDER)]
#     targets = sorted(df_5shot["target_progression"].unique())

#     fig = plt.figure(figsize=(10, 5.5))
#     gs = gridspec.GridSpec(1, 2, width_ratios=[1.6, 1], wspace=0.35)

#     # --- Panel A: Scatter Plot ---
#     ax_scatter = fig.add_subplot(gs[0])

#     for _, row in df_5shot.iterrows():
#         color = pcfg.PROGRESSION_COLORS.get(row["target_progression"], "#333333")
#         marker = pcfg.MODEL_MARKERS.get(row["model"], "o")
#         ax_scatter.scatter(
#             row["ref_tau_avg"],
#             row["f1_score"],
#             c=color,
#             marker=marker,
#             s=180 if marker not in ["P", "*"] else 220,
#             alpha=0.85,
#             edgecolor="white" if marker not in ["P", "*"] else color,
#             linewidth=1.5,
#             zorder=2,
#         )

#     rho_overall, p_overall = spearmanr(df_5shot["ref_tau_avg"], df_5shot["f1_score"])

#     ax_scatter.set_xlabel(r"Reference Manifold Quality (Avg $\tau$ on Training Set)", 
#                           fontsize=11, fontweight="bold")
#     ax_scatter.set_ylabel("Held-Out Accuracy (5-Shot F1)", 
#                           fontsize=11, fontweight="bold")
#     ax_scatter.set_title(
#         f"Manifold Quality Orders Held-Out Performance\nPooled ρ = {rho_overall:.2f}",
#         fontsize=12,
#         fontweight="bold",
#     )
#     ax_scatter.grid(True, linestyle="--", alpha=0.3)
#     sns.despine(ax=ax_scatter, trim=True)

#     # Legends
#     task_handles = [
#         Line2D([0], [0], marker="o", color="w", 
#                markerfacecolor=pcfg.PROGRESSION_COLORS.get(t, "#333333"), markersize=10, 
#                markeredgecolor="white", label=pcfg.PROGRESSION_NAMES.get(t, t))
#         for t in targets
#     ]

#     model_handles = [
#         Line2D([0], [0], marker=pcfg.MODEL_MARKERS.get(m, "o"), color="w",
#                markerfacecolor="#555555" if pcfg.MODEL_MARKERS.get(m) not in ["P", "*"] else "w",
#                markeredgecolor="#555555",
#                markersize=10, label=pcfg.MODEL_LABELS.get(m, m))
#         for m in pcfg.MODEL_ORDER
#     ]

#     legend1 = ax_scatter.legend(
#         handles=task_handles,
#         title="Target Task",
#         loc="upper left",
#         bbox_to_anchor=(0.0, -0.15),
#         ncol=2,
#         frameon=False,
#         fontsize=9,
#         title_fontsize=10,
#     )
#     ax_scatter.add_artist(legend1)

#     ax_scatter.legend(
#         handles=model_handles,
#         title="Model",
#         loc="upper left",
#         bbox_to_anchor=(0.55, -0.15),
#         ncol=2,
#         frameon=False,
#         fontsize=9,
#         title_fontsize=10,
#     )

#     # --- Panel B: Correlation Table ---
#     ax_table = fig.add_subplot(gs[1])
#     ax_table.axis("off")

#     stats_rows = []
#     for target in targets:
#         sub = df_5shot[df_5shot["target_progression"] == target]
#         rho, p = spearmanr(sub["ref_tau_avg"], sub["f1_score"])
#         display_name = pcfg.PROGRESSION_NAMES.get(target, target)
#         stats_rows.append({
#             "Target Task": display_name,
#             "Spearman ρ": f"{rho:.2f}",
#             "p-value": f"{p:.3f}" if p >= 0.001 else "<.001",
#         })

#     avg_rho = np.mean([float(r["Spearman ρ"]) for r in stats_rows])
#     stats_rows.append({
#         "Target Task": "AVERAGE",
#         "Spearman ρ": f"{avg_rho:.2f}",
#         "p-value": "—",
#     })

#     cols = ["Target Task", "Spearman ρ", "p-value"]
#     cell_text = [[row[c] for c in cols] for row in stats_rows]

#     table = ax_table.table(
#         cellText=cell_text,
#         colLabels=cols,
#         loc="center",
#         cellLoc="center",
#         bbox=[0.05, 0.3, 0.9, 0.6],
#     )
#     table.auto_set_font_size(False)
#     table.set_fontsize(10)
#     table.scale(1.2, 2.0)

#     for col in range(len(cols)):
#         table[(0, col)].set_text_props(fontweight="bold")
#         table[(0, col)].set_facecolor("#e6e6e6")

#     n_rows = len(stats_rows)
#     for col in range(len(cols)):
#         table[(n_rows, col)].set_text_props(fontweight="bold")
#         table[(n_rows, col)].set_facecolor("#f5f5f5")

#     ax_table.set_title(
#         "Rank Correlation\n(Reference τ vs Held-Out F1)",
#         fontsize=11,
#         fontweight="bold",
#         pad=10,
#     )

#     plt.subplots_adjust(bottom=0.25)

#     save_path = OUTPUT_DIR / "generalization_main_figure.png"
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.close()
#     print(f"✅ Saved: {save_path}")

#     return {"pooled_rho": rho_overall, "avg_rho": avg_rho}


# # -----------------------------------------------------------------------------
# # Supplementary: Per-Task Panels
# # -----------------------------------------------------------------------------
# def plot_per_task_panels(df):
#     """Supplementary figure showing each held-out task separately."""
#     df_5shot = df[df["n_shots"] == 5].copy()
#     df_5shot = df_5shot[df_5shot["model"].isin(pcfg.MODEL_ORDER)]
#     targets = sorted(df_5shot["target_progression"].unique())

#     fig, axes = plt.subplots(1, len(targets), figsize=(3.5 * len(targets), 4), sharey=True)

#     for idx, target in enumerate(targets):
#         ax = axes[idx]
#         sub = df_5shot[df_5shot["target_progression"] == target]

#         sns.scatterplot(
#             data=sub,
#             x="ref_tau_avg",
#             y="f1_score",
#             hue="model",
#             palette=pcfg.MODEL_COLORS,
#             hue_order=pcfg.MODEL_ORDER,
#             s=180,
#             alpha=0.85,
#             edgecolor="white",
#             linewidth=1,
#             ax=ax,
#             legend=(idx == len(targets) - 1),
#         )

#         rho, p = spearmanr(sub["ref_tau_avg"], sub["f1_score"])
#         display_name = pcfg.PROGRESSION_NAMES.get(target, target)

#         ax.set_title(f"Target: {display_name}\nρ = {rho:.2f} (p = {p:.3f})", 
#                      fontsize=11, fontweight="bold")
#         ax.set_xlabel("Reference τ", fontsize=10)
#         if idx == 0:
#             ax.set_ylabel("Held-Out 5-Shot F1", fontsize=10, fontweight="bold")
#         ax.grid(True, linestyle="--", alpha=0.3)

#     if axes[-1].get_legend():
#         axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=9)

#     plt.suptitle("Per-Task: Reference τ vs Held-Out F1", fontsize=13, fontweight="bold", y=1.02)
#     plt.tight_layout()

#     save_path = OUTPUT_DIR / "generalization_per_task_supplement.png"
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.close()
#     print(f"✅ Saved: {save_path}")


# # -----------------------------------------------------------------------------
# # Main
# # -----------------------------------------------------------------------------
# def main():
#     print("=" * 60)
#     print("PLOTTING: Generalizability Analysis")
#     print("=" * 60)

#     try:
#         gen_df, tau_df = load_data()
#     except FileNotFoundError as e:
#         print(f"Error: {e}")
#         sys.exit(1)

#     print(f"Loaded {len(gen_df)} rows")

#     # Figure 1: Same-disease τ vs F1
#     same_disease_results = plot_same_disease_scatter(gen_df, tau_df)
#     print(f"Same-disease: ρ = {same_disease_results['rho']:.2f}")

#     # Figure 2: Cross-task generalization
#     cross_task_results = plot_generalization_figure(gen_df)
#     print(f"Cross-task: pooled ρ = {cross_task_results['pooled_rho']:.2f}, avg ρ = {cross_task_results['avg_rho']:.2f}")

#     # Supplementary: Per-task panels
#     plot_per_task_panels(gen_df)

#     print("\n✅ Done!")


# if __name__ == "__main__":
#     main()