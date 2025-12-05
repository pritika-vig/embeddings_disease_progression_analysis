import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path # Use pathlib for safer directory handling

# 1. Manually input your results
data = [
    {"progression": "BDC", "model": "uni2", "tau": 0.699, "lower": 0.656, "upper": 0.733, "type": "Vision-Only"},
    {"progression": "BDC", "model": "virchow2", "tau": 0.619, "lower": 0.575, "upper": 0.663, "type": "Vision-Only"},
    {"progression": "BDC", "model": "conch", "tau": 0.598, "lower": 0.540, "upper": 0.656, "type": "VLM"},
    {"progression": "BDC", "model": "gigapath", "tau": 0.545, "lower": 0.444, "upper": 0.608, "type": "Vision-Only"},
    {"progression": "BDC", "model": "musk", "tau": 0.344, "lower": 0.290, "upper": 0.539, "type": "VLM"},
    {"progression": "BDC", "model": "dinov2", "tau": 0.217, "lower": 0.019, "upper": 0.285, "type": "Natural Image"},

    {"progression": "CRC-Conv", "model": "gigapath", "tau": 0.736, "lower": 0.723, "upper": 0.750, "type": "Vision-Only"},
    {"progression": "CRC-Conv", "model": "virchow2", "tau": 0.733, "lower": 0.722, "upper": 0.743, "type": "Vision-Only"},
    {"progression": "CRC-Conv", "model": "uni2", "tau": 0.725, "lower": 0.624, "upper": 0.741, "type": "Vision-Only"},
    {"progression": "CRC-Conv", "model": "conch", "tau": 0.703, "lower": 0.500, "upper": 0.732, "type": "VLM"},
    {"progression": "CRC-Conv", "model": "musk", "tau": 0.676, "lower": 0.460, "upper": 0.705, "type": "VLM"},
    {"progression": "CRC-Conv", "model": "dinov2", "tau": 0.517, "lower": 0.255, "upper": 0.587, "type": "Natural Image"},

    {"progression": "CRC-Serr", "model": "virchow2", "tau": 0.793, "lower": 0.786, "upper": 0.799, "type": "Vision-Only"},
    {"progression": "CRC-Serr", "model": "gigapath", "tau": 0.792, "lower": 0.779, "upper": 0.795, "type": "Vision-Only"},
    {"progression": "CRC-Serr", "model": "uni2", "tau": 0.788, "lower": 0.780, "upper": 0.792, "type": "Vision-Only"},
    {"progression": "CRC-Serr", "model": "musk", "tau": 0.705, "lower": 0.473, "upper": 0.714, "type": "VLM"},
    {"progression": "CRC-Serr", "model": "dinov2", "tau": 0.691, "lower": 0.362, "upper": 0.709, "type": "Natural Image"},
    {"progression": "CRC-Serr", "model": "conch", "tau": 0.642, "lower": 0.611, "upper": 0.695, "type": "VLM"},

    {"progression": "SCC", "model": "virchow2", "tau": 0.717, "lower": 0.707, "upper": 0.730, "type": "Vision-Only"},
    {"progression": "SCC", "model": "uni2", "tau": 0.707, "lower": 0.480, "upper": 0.724, "type": "Vision-Only"},
    {"progression": "SCC", "model": "musk", "tau": 0.641, "lower": 0.594, "upper": 0.668, "type": "VLM"},
    {"progression": "SCC", "model": "conch", "tau": 0.637, "lower": 0.592, "upper": 0.657, "type": "VLM"},
    {"progression": "SCC", "model": "gigapath", "tau": 0.590, "lower": 0.567, "upper": 0.618, "type": "Vision-Only"},
    {"progression": "SCC", "model": "dinov2", "tau": 0.476, "lower": 0.150, "upper": 0.493, "type": "Natural Image"},
]

df = pd.DataFrame(data)

# 2. Plotting (Seaborn deprecation fixes applied)
plt.figure(figsize=(12, 7))
sns.set_theme(style="whitegrid")

# Sort models by average performance
order = df.groupby("model")["tau"].mean().sort_values(ascending=False).index
palette = {"Vision-Only": "#3498db", "VLM": "#9b59b6", "Natural Image": "#95a5a6"}

# Use linestyle='none' instead of join=False
# Use err_kws instead of errwidth
ax = sns.pointplot(
    data=df, 
    x="model", 
    y="tau", 
    hue="type", 
    order=order,
    linestyle='none',    # Fixed
    dodge=0.4, 
    palette=palette,
    capsize=0.1,
    err_kws={'linewidth': 1.5} # Fixed
)

# Clear the seaborn attempt so we can overlay manual CIs
plt.clf() 

# 3. MANUAL PLOT FOR PRE-COMPUTED CIs
fig, ax = plt.subplots(figsize=(12, 6))

models = list(order)
x_base = range(len(models))
width = 0.15
offsets = {"BDC": -1.5*width, "CRC-Conv": -0.5*width, "CRC-Serr": 0.5*width, "SCC": 1.5*width}
colors = {"BDC": "#e74c3c", "CRC-Conv": "#3498db", "CRC-Serr": "#2ecc71", "SCC": "#f1c40f"}

for prog, group in df.groupby("progression"):
    group = group.set_index("model").reindex(models)
    
    xs = [x_base[i] + offsets[prog] for i in range(len(models))]
    ys = group["tau"].values
    
    # Matplotlib errorbar expects relative errors (val - lower, upper - val)
    yerr = [
        (group["tau"] - group["lower"]).values, 
        (group["upper"] - group["tau"]).values
    ]
    
    ax.errorbar(
        xs, ys, yerr=yerr, 
        fmt='o', label=prog, color=colors[prog], 
        capsize=3, elinewidth=1.5, alpha=0.8
    )

ax.set_xticks(x_base)
ax.set_xticklabels(models, fontsize=11, fontweight='bold')
ax.set_ylabel("Kendall's Tau (Trajectory Fidelity)", fontsize=12)
ax.set_title("Emergence of Temporal Structure across Foundation Models", fontsize=14)
ax.legend(title="Disease Progression")
ax.grid(True, axis='y', linestyle='--', alpha=0.5)

# Background shading
ax.axvspan(-0.5, 2.5, color='blue', alpha=0.05, label='_nolegend_')
ax.text(1, 0.1, "Vision-Only (Pathology)", ha='center', color='blue', alpha=0.5, fontweight='bold')

ax.axvspan(2.5, 4.5, color='purple', alpha=0.05, label='_nolegend_')
ax.text(3.5, 0.1, "Vision-Language", ha='center', color='purple', alpha=0.5, fontweight='bold')

ax.axvspan(4.5, 5.5, color='gray', alpha=0.05, label='_nolegend_')
ax.text(5, 0.1, "Natural Image", ha='center', color='gray', alpha=0.5, fontweight='bold')

plt.tight_layout()

# 4. FIX: CREATE DIRECTORY BEFORE SAVING
output_path = Path("studies/01_emergence_of_time/plots/model_comparison_v2.png")
output_path.parent.mkdir(parents=True, exist_ok=True) # <--- This creates the folder

plt.savefig(output_path, dpi=300)
print(f"✅ Plot successfully saved to {output_path}")