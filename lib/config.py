import os
import datetime
import numpy as np
import matplotlib.colors as mcolors

# ================= INPUT / OUTPUT =================
DATA_DIR = '/home/oliver/Documents/MET/VIIRS_Mask/20250415_noaa21-viirs-pps'
OUTPUT_DIR = '/home/oliver/Documents/MET/VIIRS_Mask/output_fram_strait'
DEBUG_DIR = os.path.join(OUTPUT_DIR, 'debug_maps')

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
PDF_FILENAME = f"VIIRS_FramStrait_{TIMESTAMP}.pdf"

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
if not os.path.exists(DEBUG_DIR): os.makedirs(DEBUG_DIR)

# ================= GEOGRAPHIC CROP (FRAM STRAIT) =================
USE_CROP = True

# Fram Strait / Svalbard Area
CROP_BOUNDS = {
    'lat_min': 75.0,
    'lat_max': 82.0,
    'lon_min': -20.0,
    'lon_max': 20.0
}

# ================= PROCESSING SETTINGS =================
# Latitude Threshold (Ignored if USE_CROP is True)
LATITUDE_THRESHOLD = 60.0

# Solar Zenith Limit: 85.0 = Day only
MAX_SOLAR_ZENITH = 85.0 

MIN_ICE_PIXELS = 200
NUM_WORKERS = None 
MOSAIC_STEP = 40
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
COLORS_LIST = ['black', 'blue', 'white', 'silver', 'green']
MOSAIC_CMAP = mcolors.ListedColormap(COLORS_LIST)
MOSAIC_NORM = mcolors.BoundaryNorm([0, 1, 2, 3, 4, 5], 5)

NEIGHBOR_COLORS = {
    'iNw': 'red',      'i2Nw': 'orange', 'i3Nw': 'gold',
    'wNi': 'cyan',     'w2Ni': 'navy',   'w3Ni': 'purple',
    'iNc': 'lightgray','i2Nc': 'lightslategray',
    'wNc': 'lightgray','w2Nc': 'lightslategray',
    'Mixed': 'orchid'
}