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
    'gigapath': 'Prov-GigaPath'
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

# Full titles for figure headers
PROGRESSION_TITLES = {
    "SCC": "Cutaneous Squamous Cell Carcinoma",
    "BDC": "Breast Ductal Carcinoma", 
    "CRC-Conventional": "Colorectal Adenoma-Carcinoma (Conventional)",
    "CRC-Serrated": "Colorectal Adenoma-Carcinoma (Serrated)",
}

# Maps full class names to abbreviated display names
CLASS_SHORT_NAMES = {
    # BDC (Breast Ductal Carcinoma)
    "Ductal carcinoma in situ (low-grade)": "DCIS\n(Low Grade)",
    "Ductal carcinoma in situ (high-grade)": "DCIS\n(High Grade)",
    "Invasive non-special type carcinoma": "Invasive\nCarcinoma",
    
    # SCC (Squamous Cell Carcinoma)
    "Epidermis": "Epidermis",
    "Actinic keratosis": "Actinic\nKeratosis",
    "Carcinoma in situ": "Carcinoma\nin Situ",
    "Squamous cell carcinoma": "SCC",
    
    # CRC-Conventional (Colorectal - Conventional Pathway)
    "Adenoma low grade": "Adenoma\n(Low)",
    "Adenoma high grade": "Adenoma\n(High)",
    "Adenocarcinoma low grade": "Adenoca\n(Low)",
    "Adenocarcinoma high grade": "Adenoca\n(High)",
    
    # CRC-Serrated (Colorectal - Serrated Pathway)
    "Hyperplastic polyp": "Hyperplastic\nPolyp",
    "Sessile serrated lesion": "Sessile\nSerrated",
    # "Adenocarcinoma high grade" already defined above
}

# Single-line versions for inline text or tight spaces
CLASS_SHORT_NAMES_INLINE = {
    # BDC
    "Ductal carcinoma in situ (low-grade)": "DCIS (Low)",
    "Ductal carcinoma in situ (high-grade)": "DCIS (High)",
    "Invasive non-special type carcinoma": "Invasive Ca",
    
    # SCC
    "Epidermis": "Epidermis",
    "Actinic keratosis": "Actinic Keratosis",
    "Carcinoma in situ": "Ca in Situ",
    "Squamous cell carcinoma": "SCC",
    
    # CRC-Conventional
    "Adenoma low grade": "Adenoma (Low)",
    "Adenoma high grade": "Adenoma (High)",
    "Adenocarcinoma low grade": "Adenoca (Low)",
    "Adenocarcinoma high grade": "Adenoca (High)",
    
    # CRC-Serrated
    "Hyperplastic polyp": "HP",
    "Sessile serrated lesion": "SSL",
}


def get_class_short_name(class_name: str, inline: bool = False) -> str:
    """
    Get the short display name for a class.
    
    Args:
        class_name: Full class name from config
        inline: If True, returns single-line version; otherwise multi-line
        
    Returns:
        Short display name, or original name if not found
    """
    lookup = CLASS_SHORT_NAMES_INLINE if inline else CLASS_SHORT_NAMES
    return lookup.get(class_name, class_name)


def get_progression_title(prog_name: str) -> str:
    """Get a human-readable title for a progression."""
    return PROGRESSION_TITLES.get(prog_name, prog_name)


# -----------------------------------------------------------------------------
# 2. Color Palettes
# -----------------------------------------------------------------------------

# Standard colors for diseases (Progressions)
PROGRESSION_COLORS = {
    "BDC": "#D62728",               # Red
    "CRC-Conventional": "#1F77B4",  # Blue
    "CRC-Serrated": "#2CA02C",      # Green
    "SCC": "#FF7F0E",               # Orange
    "Null": "#4a4a4a"               # Gray
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

CATEGORY_LABEL_COLORS = {
    'natural': '#4a4a4a',           # Dark gray
    'vision_language': '#6b3fa0',   # Darker purple
    'vision_only': '#1a5276',       # Darker blue
}

# -----------------------------------------------------------------------------
# 3. ICML Styling Function
# -----------------------------------------------------------------------------

# ICML-optimized font sizes (for figures)
# FONT_SIZES = {
#     'title': 18,
#     'subtitle': 13,
#     'section_header': 13,
#     'section_subheader': 10,
#     'plot_title': 14,
#     'axis_label': 11,
#     'tick_label': 10,
#     'legend': 10,
#     'annotation': 9,
#     'small': 10,
# }

FONT_SIZES = {
    'title': 18,
    'subtitle': 14,
    'section_header': 14,
    'section_subheader': 11,
    'plot_title': 16,
    'axis_label': 13,
    'tick_label': 12,
    'legend': 11,
    'annotation': 11,
    'small': 10,
    'category_label': 12,  
}

def set_icml_style():
    """Applies ICML-compliant matplotlib style settings."""
    sns.set_theme(style="white", context="paper", font_scale=1.2)
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'figure.dpi': 300
    })