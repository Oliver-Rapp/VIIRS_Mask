import os
import datetime
import numpy as np
import matplotlib.colors as mcolors

# ================= PATHS =================
DATA_DIR = '/home/oliver/Documents/MET/VIIRS_Mask/data-case1'
OUTPUT_DIR = '/home/oliver/Documents/MET/VIIRS_Mask/output_multi_scene'
DEBUG_DIR = os.path.join(OUTPUT_DIR, 'debug_maps')

if not os.path.exists(DEBUG_DIR):
    os.makedirs(DEBUG_DIR)

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
PDF_FILENAME = f"VIIRS_Report_{TIMESTAMP}.pdf"

# ================= RUN SETTINGS =================
NUM_WORKERS = None           
NUM_DEBUG_MAPS = 100        
Example_Run = False       


# ================= CROP & THRESHOLDS =================
USE_CROP = True
CROP_BOUNDS = {
    'lat_min': 76.192, 'lat_max': 77.918,
    'lon_min': 13.789, 'lon_max': 18.63
}
LATITUDE_THRESHOLD = 60.0
MAX_SOLAR_ZENITH = 85.0
MIN_ICE_PIXELS = 50 

# ================= COVERAGE MAP SETTINGS =================
# Resolution in degrees for the heatmap (0.1 deg is approx 10km grid)
MAP_RES = 0.1 

# Generate fixed bin edges for the heatmap based on CROP_BOUNDS
# We add a small buffer to ensure edges are caught
COVERAGE_LON_BINS = np.arange(
    CROP_BOUNDS['lon_min'] - 1, 
    CROP_BOUNDS['lon_max'] + 1, 
    MAP_RES
)
COVERAGE_LAT_BINS = np.arange(
    CROP_BOUNDS['lat_min'] - 1, 
    CROP_BOUNDS['lat_max'] + 1, 
    MAP_RES
)

# ================= HISTOGRAM CONFIGURATION =================
# BINS: Defined broadly to capture data
# XLIMS: Defined strictly to match the visual zoom of the reference images

BINS = {
    't11': np.linspace(220, 300, 161),       
    'diff1': np.linspace(-2, 5, 141),       # T11 - T12
    'diff2': np.linspace(-5, 15, 201),      # T3.7 - T12
    'diff3': np.linspace(-5, 5, 101),       # T8.7 - T12
    'sunz': np.linspace(0, 90, 91),
    'satz': np.linspace(0, 70, 71),
    # Textures: High resolution bins for smooth curves
    'r06_tex': np.linspace(0, 40, 201),      
    't11_tex': np.linspace(0, 5, 101),       
    't11t12_tex': np.linspace(0, 1.0, 101),  
    't37t12_tex': np.linspace(0, 8, 161)     
}

# EXACT PLOTTING LIMITS (from reference images)
XLIMS = {
    't11': (255, 275),
    'diff1': (0, 2),
    'diff2': (0, 11),
    'diff3': (-2.5, 1),
    't11t12_tex': (0, 0.3),
    't37t12_tex': (0, 4)
    # r06_tex, sunz, satz, t11_tex: Use full range or auto
}

# ================= PLOTTING DEFINITIONS =================
KEYS = ['iNw', 'i2Nw', 'i3Nw', 'wNi', 'w2Ni', 'w3Ni', 'iNc', 'i2Nc', 'wNc', 'w2Nc', 'Mixed']

# Maps internal keys to the styling used in the original script
LABEL_MAP = {
    'iNw': 'InW', 'i2Nw': 'I2nW', 'i3Nw': 'I3nW',
    'wNi': 'WnI', 'w2Ni': 'W2nI', 'w3Ni': 'W3nI',
    'iNc': 'InC', 'i2Nc': 'I2nC',
    'wNc': 'WnC', 'w2Nc': 'W2nC',
    'Mixed': 'Mixed'
}

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
COLORS_LIST = ['black', 'blue', 'white', 'silver', 'green']
MOSAIC_CMAP = mcolors.ListedColormap(COLORS_LIST)
MOSAIC_NORM = mcolors.BoundaryNorm([0, 1, 2, 3, 4, 5], 5)

NEIGHBOR_COLORS = {
    # Ice -> Water Edge (Warm colors)
    'iNw': 'red',      
    'i2Nw': 'darkorange', 
    'i3Nw': 'gold',

    # Water -> Ice Edge (Cool Blues)
    'wNi': 'cyan',     
    'w2Ni': 'dodgerblue',   
    'w3Ni': 'navy',

    # Cloud Neighbors (Distinguishable colors)
    # Ice-side Cloud neighbors (Purples)
    'iNc': 'mediumorchid',   
    'i2Nc': 'darkviolet',
    
    # Water-side Cloud neighbors (Teals/Greens)
    'wNc': 'mediumseagreen', 
    'w2Nc': 'darkgreen',

    # Mixed
    'Mixed': 'deeppink'
}