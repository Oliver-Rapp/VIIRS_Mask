import os
import numpy as np
import matplotlib.colors as mcolors

# ================= INPUT / OUTPUT =================
DATA_DIR = '/home/oliver/Documents/MET/VIIRS_Mask/20250415_noaa21-viirs-pps'
OUTPUT_DIR = '/home/oliver/Documents/MET/VIIRS_Mask/output_day_stats_final'
PDF_FILENAME = 'VIIRS_Daily_Report.pdf'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ================= PROCESSING SETTINGS =================
# Latitude Threshold: Skip scenes completely below this latitude
LATITUDE_THRESHOLD = 60.0

# Minimum Ice Pixels: Skip scene if it contains fewer than this many ice pixels
MIN_ICE_PIXELS = 200

# Multiprocessing: Set to None to use all CPU cores
NUM_WORKERS = None 

# Mosaic Subsampling: Take every Nth pixel for the map
MOSAIC_STEP = 40

# --- DEBUG MAPPING ---
# Attempt to plot this many of the initial files found.
NUM_DEBUG_MAPS = 50 

# ================= HISTOGRAM BINS =================
BINS = {
    't11': np.linspace(240, 280, 81),       
    'diff1': np.linspace(0, 3, 61),         
    'diff2': np.linspace(0, 12, 61),        
    'diff3': np.linspace(-3, 1.5, 91),      
    'sunz': np.linspace(0, 90, 91),
    'satz': np.linspace(0, 70, 71),
    'r06_tex': np.linspace(0, 40, 81),      
    't11_tex': np.linspace(0, 5, 51),       
    't11t12_tex': np.linspace(0, 0.5, 51),  
    't37t12_tex': np.linspace(0, 6, 61)     
}

# Strict Plotting Limits (X-Axis)
XLIMS = {
    't11': (255, 275),
    'diff1': (0, 2),
    'diff2': (0, 11),
    'diff3': (-2.5, 1),
    't11t12_tex': (0, 0.3),
    't37t12_tex': (0, 4)
}

# ================= DEFINITIONS =================
KEYS = ['iNw', 'i2Nw', 'i3Nw', 'wNi', 'w2Ni', 'w3Ni', 'iNc', 'i2Nc', 'wNc', 'w2Nc', 'Mixed']

PLOT_GROUPS = {
    'Ice Neighbors':   ['iNw', 'i2Nw', 'i3Nw'],
    'Water Neighbors': ['wNi', 'w2Ni', 'w3Ni'],
    'Cloud Neighbors': ['iNc', 'i2Nc', 'wNc', 'w2Nc'],
    'Mixed Neighbors': ['Mixed']
}

PLOT_ORDER = [
    ('t11', 'T11 (K)'),
    ('diff1', 'T11 - T12 (K)'),
    ('diff2', 'T37 - T12 (K)'),
    ('diff3', 'T8.7 - T12 (K)'),
    ('sunz', 'Sun Zenith'),
    ('satz', 'Sat Zenith'),
    ('r06_tex', 'r06-texture'),
    ('t11_tex', 't11-texture'),
    ('t11t12_tex', 't11t12-texture'),
    ('t37t12_tex', 't37t12-texture')
]

# ================= COLORS =================
# Define Discrete Colors
# 0=No Data (Black), 1=Water (Blue), 2=Ice (White), 3=Cloud (Silver), 4=Land (Green)
COLORS_LIST = ['black', 'blue', 'white', 'silver', 'green']

# Create Colormap object using the module alias
MOSAIC_CMAP = mcolors.ListedColormap(COLORS_LIST)

# Define boundaries so values fall exactly into the colors
# 0->Black, 1->Blue, 2->White, 3->Silver, 4->Green
MOSAIC_NORM = mcolors.BoundaryNorm([0, 1, 2, 3, 4, 5], 5)