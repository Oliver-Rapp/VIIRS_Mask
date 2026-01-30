import os
import datetime
import numpy as np
import matplotlib.colors as mcolors

# ================= PATHS =================
DATA_DIR = '/home/oliver/Documents/MET/VIIRS_Mask/data/20250403_noaa21-viirs-pps'
OUTPUT_DIR = '/home/oliver/Documents/MET/VIIRS_Mask/output_multi_scene'

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ================= REGION SELECTION =================
# Define available bounding boxes here
PREDEFINED_REGIONS = {
    'svalbard': {
        'lat_min': 74.0, 'lat_max': 81.0,
        'lon_min': 10.0, 'lon_max': 35.0
    },
    'barents_and_fram': {
        # The original broad bounds
        'lat_min': 67.801, 'lat_max': 82.792,
        'lon_min': -33.363, 'lon_max': 76.105
    },
    'barents_sea': {
        'lat_min': 70.0, 'lat_max': 78.0,
        'lon_min': 15.0, 'lon_max': 55.0
    }
}

# --- SELECT YOUR REGION HERE ---
SELECTED_REGION = 'barents_and_fram' 

# Set the active bounds based on selection
if SELECTED_REGION not in PREDEFINED_REGIONS:
    raise ValueError(f"Region '{SELECTED_REGION}' not found in PREDEFINED_REGIONS.")

CROP_BOUNDS = PREDEFINED_REGIONS[SELECTED_REGION]
USE_CROP = True

# Update Filename to include Region Name
PDF_FILENAME = f"VIIRS_Report_{SELECTED_REGION}_{TIMESTAMP}.pdf"

# ================= DEBUG DIRS =================
DEBUG_BASE = os.path.join(OUTPUT_DIR, 'debug_maps')
DEBUG_DIR = os.path.join(DEBUG_BASE, TIMESTAMP)

if not os.path.exists(DEBUG_DIR):
    os.makedirs(DEBUG_DIR)

# ================= RUN SETTINGS =================
NUM_WORKERS = None           
NUM_DEBUG_MAPS = 100        

# --- MIZ SUBSET SETTINGS ---
# Generate a second set of histograms filtered by T11 temperature
ENABLE_MIZ_HISTOGRAMS = True
MIZ_T11_RANGE = (268.5, 271.5) # Kelvin

# ================= THRESHOLDS =================
LATITUDE_THRESHOLD = 60.0
MAX_SOLAR_ZENITH = 85.0
MIN_ICE_PIXELS = 50 

# ================= COVERAGE MAP SETTINGS =================
MAP_RES = 0.1 
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
BINS = {
    't11': np.linspace(220, 300, 161),       
    'diff1': np.linspace(-2, 5, 141),       
    'diff2': np.linspace(-5, 15, 201),      
    'diff3': np.linspace(-5, 5, 101),       
    'sunz': np.linspace(0, 90, 91),
    'satz': np.linspace(0, 70, 71),
    'r06_tex': np.linspace(0, 40, 201),      
    't11_tex': np.linspace(0, 5, 101),       
    't11t12_tex': np.linspace(0, 1.0, 101),  
    't37t12_tex': np.linspace(0, 8, 161)     
}

XLIMS = {
    't11': (255, 275),
    'diff1': (0, 2),
    'diff2': (0, 11),
    'diff3': (-2.5, 1),
    't11t12_tex': (0, 0.3),
    't37t12_tex': (0, 4)
}

# ================= PLOTTING DEFINITIONS =================
KEYS = [
    # Ice Neighbors
    'InW', 'I2nW', 'I3nW', 
    'InC', 'I2nC', 'I3nC',
    
    # Water Neighbors
    'WnI', 'W2nI', 'W3nI', 
    'WnC', 'W2nC', 'W3nC',
    
    # Cloud Neighbors
    'CnI', 'C2nI', 'C3nI',
    'CnW', 'C2nW', 'C3nW',

    # Interiors
    'IN', 'WN', 'CN',
    
    # Complex
    'Mixed'
]

# Labels match the keys exactly as requested
LABEL_MAP = {k: k for k in KEYS}

PLOT_GROUPS = {
    'Ice Analysis':   ['IN', 'InW', 'InC'],
    'Water Analysis': ['WN', 'WnI', 'WnC'],
    'Cloud Analysis': ['CN', 'CnI', 'CnW'], 
    'Ice vs Water Edges': ['InW', 'WnI', 'IN', 'WN'],
    'Mixed vs Pure': ['Mixed', 'IN', 'WN']
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
    # --- Ice Base: DarkViolet ---
    'IN': 'darkviolet',
    # Ice -> Water (Red/Orange)
    'InW': 'red', 'I2nW': 'darkorange', 'I3nW': 'gold',
    # Ice -> Cloud (Purple/Pink)
    'InC': 'mediumorchid', 'I2nC': 'orchid', 'I3nC': 'thistle',

    # --- Water Base: Blue ---
    'WN': 'blue',
    # Water -> Ice (Cyan/Teal)
    'WnI': 'cyan', 'W2nI': 'deepskyblue', 'W3nI': 'dodgerblue',
    # Water -> Cloud (Green)
    'WnC': 'limegreen', 'W2nC': 'mediumseagreen', 'W3nC': 'seagreen',

    # --- Cloud Base: DimGray ---
    'CN': 'dimgray',
    # Cloud -> Ice (Warm Pinks)
    'CnI': 'hotpink', 'C2nI': 'palevioletred', 'C3nI': 'pink',
    # Cloud -> Water (Light Blues)
    'CnW': 'lightskyblue', 'C2nW': 'lightblue', 'C3nW': 'aliceblue',

    # --- Complex ---
    'Mixed': 'deeppink' 
}