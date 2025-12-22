import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# 1. Model Standards (Order & Metadata)
# -----------------------------------------------------------------------------

# Desired Order: Natural -> Vision-Language -> Vision-Only (Pathology)
MODEL_ORDER = [
    'dinov2',       # Natural
    'conch',        # VL
    'musk',         # VL
    'uni2',         # Path
    'virchow2',     # Path
    'gigapath'      # Path
]

MODEL_LABELS = {
    'dinov2': 'DINOv2',
    'conch': 'CONCH',
    'musk': 'MuSK',
    'uni2': 'UNI-2',
    'virchow2': 'Virchow-2',
    'gigapath': 'GigaPath'
}

MODEL_TYPE_MAPPING = {
    'dinov2': 'DINOv2 Baseline',
    'conch': 'Vision-Language (Path)',
    'musk': 'Vision-Language (Path)',
    'uni2': 'Vision-Only (Path)',
    'virchow2': 'Vision-Only (Path)',
    'gigapath': 'Vision-Only (Path)'
}

PROGRESSION_NAMES= {
    "BDC": "BDC",        
    "CRC-Conventional": "CRC-Conv",  
    "CRC-Serrated": "CRC-Serr",      
    "SCC": "SCC",         
}


# -----------------------------------------------------------------------------
# 2. Color Palettes
# -----------------------------------------------------------------------------

# Standard colors for diseases (Progressions)
PROGRESSION_COLORS = {
    "BDC": "#D62728",               # Red
    "CRC-Conventional": "#1F77B4",  # Blue
    "CRC-Serrated": "#2CA02C",      # Green
    "SCC": "#FF7F0E",               # Orange
    "Null": "#7f8c8d"               # Gray
}

# If you need specific colors per model (e.g. for bar charts)
MODEL_COLORS = {
    'dinov2': '#7f7f7f',    # Grey for natural
    'conch': '#9467bd',     # Purple for VL
    'musk': '#8c564b',      # Brown for VL
    'uni2': '#17becf',      # Cyan for Path
    'virchow2': '#e377c2',  # Pink for Path
    'gigapath': '#bcbd22'   # Olive for Path
}

# Specific markers per model
# Logic: 
# - Circle 'o' for Natural (Baseline)
# - Square 's' / Diamond 'D' for VL (Blocky/Structural)
# - Star/Plus/Triangle for Vision-Only (High Performance)
MODEL_MARKERS = {
    'dinov2': 'o',      # Circle
    'conch': 's',       # Square
    'musk': 'D',        # Diamond
    'uni2': 'P',        # Plus (Filled)
    'virchow2': '*',    # Star
    'gigapath': '^'     # Triangle Up
}

# -----------------------------------------------------------------------------
# 3. ICML Styling Function
# -----------------------------------------------------------------------------

def set_icml_style():
    """Applies ICML-compliant matplotlib style settings."""
    sns.set_theme(style="white", context="paper", font_scale=1.2)
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'figure.dpi': 300
    })