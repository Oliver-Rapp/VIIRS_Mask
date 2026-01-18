import os
import re
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
from scipy import ndimage
from netCDF4 import Dataset

# ================= CONFIGURATION =================
DATA_DIR = '/home/oliver/Documents/MET/VIIRS_Mask/20250415_noaa21-viirs-pps'
OUTPUT_DIR = '/home/oliver/Documents/MET/VIIRS_Mask/output_day_stats_full'
PDF_FILENAME = 'VIIRS_Analysis_Report.pdf'
LATITUDE_THRESHOLD = 60.0
SAVE_DEBUG_MAP_PNG = True  # Save the first scene as a standalone PNG as well?

# Ensure output exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ================= COLORS & STYLES =================
colors = [
    'black', 'blue', 'white', 'gray', 'green', 
    'red', 'orange', 'gold',                   
    'cyan', 'navy', 'purple'                   
]
labels = ['No data', 'Water', 'Ice', 'Clouds', 'Land', 'InW', 'I2nW', 'I3nW', 'WnI', 'W2nI', 'W3nI']
cmap = ListedColormap(colors)

# ================= ACCUMULATOR CLASS =================
class HistogramAccumulator:
    def __init__(self, name, bins, xlim=None):
        self.name = name
        self.bins = bins
        self.xlim = xlim
        self.keys = [
            'iNw', 'i2Nw', 'i3Nw',
            'wNi', 'w2Ni', 'w3Ni',
            'iNc', 'i2Nc', 'wNc', 'w2Nc',
            'Mixed'
        ]
        self.counts = {k: np.zeros(len(bins)-1) for k in self.keys}

    def add(self, key, data_values):
        if len(data_values) == 0: return
        if np.ma.is_masked(data_values):
            data_values = data_values.compressed()
        hist, _ = np.histogram(data_values, bins=self.bins)
        self.counts[key] += hist

    def plot_group_to_pdf(self, pdf, group_keys, group_name):
        """Adds a plot page to the open PDF object."""
        plt.figure(figsize=(10, 6))
        
        has_data = False
        bin_centers = (self.bins[:-1] + self.bins[1:]) / 2
        
        for key in group_keys:
            if key not in self.counts: continue
            count_array = self.counts[key]
            total = np.sum(count_array)
            if total > 0:
                has_data = True
                density = count_array / total
                plt.plot(bin_centers, density, label=f"{key} (n={int(total)})", linewidth=2)

        plt.xlabel(self.name)
        plt.ylabel("Density")
        plt.title(f"Histogram of {self.name} ({group_name})")
        if self.xlim:
            plt.xlim(self.xlim)
        
        if has_data:
            plt.legend()
        else:
            plt.text(0.5, 0.5, "No Data Processed", ha='center', transform=plt.gca().transAxes)
            
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        
        # Save to PDF instead of file
        pdf.savefig() 
        plt.close()

# ================= HELPER FUNCTIONS =================
def compute_all_neighbors(classification, cloud_bool):
    struct3 = ndimage.generate_binary_structure(2, 1)
    ice_bool = (classification == 2)
    water_bool = (classification == 1)

    # Ice-side
    dilated_water = ndimage.binary_dilation(water_bool, structure=struct3)
    iNw   = dilated_water & ice_bool & (~water_bool)
    i2Nw  = ndimage.binary_dilation(iNw, structure=struct3) & ice_bool & (~iNw)
    i3Nw = ndimage.binary_dilation(i2Nw, structure=struct3) & ice_bool & (~iNw) & (~i2Nw)

    # Water-side
    dilated_ice = ndimage.binary_dilation(ice_bool, structure=struct3)
    wNi   = dilated_ice & water_bool & (~ice_bool)
    w2Ni  = ndimage.binary_dilation(wNi, structure=struct3) & water_bool & (~wNi)
    w3Ni = ndimage.binary_dilation(w2Ni, structure=struct3) & water_bool & (~wNi) & (~w2Ni)

    # Cloud neighbors
    iNc  = ndimage.binary_dilation(cloud_bool, structure=struct3) & ice_bool
    i2Nc = ndimage.binary_dilation(iNc, structure=struct3) & ice_bool & (~iNc)
    wNc  = ndimage.binary_dilation(cloud_bool, structure=struct3) & water_bool
    w2Nc = ndimage.binary_dilation(wNc, structure=struct3) & water_bool & (~wNc)

    return {
        'iNw': iNw, 'i2Nw': i2Nw, 'i3Nw': i3Nw,
        'wNi': wNi, 'w2Ni': w2Ni, 'w3Ni': w3Ni,
        'iNc': iNc, 'i2Nc': i2Nc, 'wNc': wNc, 'w2Nc': w2Nc
    }

def get_file_groups(directory):
    groups = {}
    files = [f for f in os.listdir(directory) if f.endswith('.nc')]
    for f in files:
        match = re.search(r'_(\d{8}T\d{7}Z)_', f) 
        if not match: continue
        ts = match.group(1)
        if ts not in groups: groups[ts] = {}
        path = os.path.join(directory, f)
        if 'S_NWC_CMA_' in f: groups[ts]['cma'] = path
        elif 'physiography' in f: groups[ts]['geo'] = path
        elif 'viirs' in f: groups[ts]['l1b'] = path
        elif 'textures' in f: groups[ts]['tex'] = path
    return groups

# ================= INITIALIZATION =================
accs = {}
# 1. Channels
accs['t11'] = HistogramAccumulator('T11 (K)', np.linspace(230, 290, 121), xlim=(240, 280))
# 2. Differences
accs['diff1'] = HistogramAccumulator('T11 - T12 (K)', np.linspace(-1, 5, 121), xlim=(0, 2.5))
accs['diff2'] = HistogramAccumulator('T3.7 - T12 (K)', np.linspace(-5, 15, 101), xlim=(0, 11))
accs['diff3'] = HistogramAccumulator('T8.7 - T12 (K)', np.linspace(-4, 4, 81), xlim=(-2.5, 1))
# 3. Geometry
accs['sunz'] = HistogramAccumulator('Sun Zenith', np.linspace(0, 90, 91))
accs['satz'] = HistogramAccumulator('Sat Zenith', np.linspace(0, 70, 71))
# 4. Textures
accs['r06_tex'] = HistogramAccumulator('r06-texture', np.linspace(0, 10, 101))
accs['t11_tex'] = HistogramAccumulator('t11-texture', np.linspace(0, 5, 101))
accs['t11t12_tex'] = HistogramAccumulator('t11t12-texture', np.linspace(0, 1, 101), xlim=(0, 0.3))
accs['t37t12_tex'] = HistogramAccumulator('t37t12-texture', np.linspace(0, 8, 101), xlim=(0, 6))

# Lists to track coverage for the PDF map
valid_lat_centers = []
valid_lon_centers = []

# ================= PROCESSING LOOP =================
file_groups = get_file_groups(DATA_DIR)
print(f"Found {len(file_groups)} scenes.")
count_processed = 0

# Capture start time
start_time = datetime.datetime.now()

# For the PDF, we will save the debug map plots to this list temporarily
debug_map_figs = [] 

for ts, files in file_groups.items():
    if not all(k in files for k in ['cma', 'geo', 'l1b', 'tex']): continue

    ds_l1b = None; ds_cma = None; ds_geo = None; ds_tex = None

    try:
        # 1. Latitude & Longitude Check
        ds_l1b = Dataset(files['l1b'], 'r')
        if 'lat' not in ds_l1b.variables: continue
        
        # Read nav
        lat = np.squeeze(ds_l1b.variables['lat'][:])
        
        # Try finding longitude (needed for coverage map)
        lon_var = None
        for k in ['lon', 'longitude', 'nav_lon']:
            if k in ds_l1b.variables:
                lon_var = k
                break
        
        # If we can't find coords or it's too far south, skip
        if np.max(lat) < LATITUDE_THRESHOLD or lon_var is None: 
            continue

        lon = np.squeeze(ds_l1b.variables[lon_var][:])

        # 2. Load Satellite Data
        t11 = ds_l1b.variables['image3'][0, :, :]
        t12 = ds_l1b.variables['image4'][0, :, :]
        t37 = ds_l1b.variables['image5'][0, :, :]
        t87 = ds_l1b.variables['image7'][0, :, :]
        sunz = ds_l1b.variables['sunzenith'][0, :, :]
        satz = ds_l1b.variables['satzenith'][0, :, :]

        # 3. Load Masks & Textures
        ds_cma = Dataset(files['cma'], 'r')
        ds_geo = Dataset(files['geo'], 'r')
        ds_tex = Dataset(files['tex'], 'r')

        cma = ds_cma.variables['cma_extended'][0, :, :]
        landuse = ds_geo.variables['landuse'][0, :, :]
        
        tex_data = {
            'r06_tex': ds_tex.variables['r06'][0,:,:].astype(float),
            't11_tex': ds_tex.variables['t11'][0,:,:].astype(float),
            't11t12_tex': ds_tex.variables['t11t12'][0,:,:].astype(float),
            't37t12_tex': ds_tex.variables['t37t12'][0,:,:].astype(float)
        }

        # 4. Generate Masks
        lat_mask = (lat >= LATITUDE_THRESHOLD)
        is_land = (landuse != 16)
        
        cloud_raw = ((cma == 1) | (cma == 2))
        cloud_clean = cloud_raw.copy()
        cloud_clean[is_land] = False
        
        classification = np.zeros_like(cma, dtype=np.uint8)
        classification[(landuse == 16) & (cma == 0)] = 1 
        classification[(cma == 3)] = 2                   
        classification[cloud_clean] = 3                  
        classification[~lat_mask] = 0

        # Optimization: Skip if little ice
        if np.count_nonzero(classification == 2) < 200: continue

        # --- DATA IS VALID ---
        # Save center point for coverage map
        valid_lat_centers.append(np.mean(lat))
        valid_lon_centers.append(np.mean(lon))

        # 5. Compute Neighbors
        nbs = compute_all_neighbors(classification, cloud_clean)
        
        # 6. Compute Mixed
        neighbor_sum = np.zeros_like(classification, dtype=int)
        for k in ['iNw','i2Nw','wNi','w2Ni','iNc','i2Nc','wNc','w2Nc']:
            if k in nbs: neighbor_sum += nbs[k].astype(int)
        nbs['Mixed'] = (neighbor_sum >= 2)

        # 7. Accumulate Data
        diffs = {'diff1': t11-t12, 'diff2': t37-t12, 'diff3': t87-t12}

        for key in accs['t11'].keys:
            if key not in nbs: continue
            mask = nbs[key]
            if not np.any(mask): continue

            accs['t11'].add(key, t11[mask])
            for dname, dval in diffs.items(): accs[dname].add(key, dval[mask])
            accs['sunz'].add(key, sunz[mask])
            accs['satz'].add(key, satz[mask])
            for tname, tval in tex_data.items(): accs[tname].add(key, tval[mask])

        # 8. Generate Debug Map (Only for first scene)
        if count_processed == 0:
            print(f"Generating Debug Map for {ts}...")
            # We create the figures but don't save yet; we keep them for the PDF
            fig_debug, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(classification, interpolation='none', cmap=cmap, vmin=0, vmax=10)
            ax.set_title(f"Sample Scene Classification: {ts}")
            ax.axis('off')
            
            # Simple overlays for debug map
            y, x = np.where(nbs['iNw'])
            ax.scatter(x, y, c='red', s=1, label='iNw')
            y, x = np.where(nbs['Mixed'])
            ax.scatter(x, y, c='orchid', s=1, label='Mixed')
            ax.legend(loc='lower right')
            
            debug_map_figs.append(fig_debug)
            
            # Optionally save standalone PNG as before
            if SAVE_DEBUG_MAP_PNG:
                fig_debug.savefig(os.path.join(OUTPUT_DIR, "debug_map_sample.png"))

        count_processed += 1
        if count_processed % 10 == 0: print(f"Processed {count_processed}...")

    except Exception as e:
        print(f"Error {ts}: {e}")
    finally:
        for d in [ds_l1b, ds_cma, ds_geo, ds_tex]:
            try: 
                if d: d.close()
            except: pass

# ================= REPORT GENERATION =================
print("Creating PDF Report...")
pdf_path = os.path.join(OUTPUT_DIR, PDF_FILENAME)

with PdfPages(pdf_path) as pdf:
    # --- PAGE 1: METADATA & COVERAGE MAP ---
    fig = plt.figure(figsize=(11, 8))
    
    # Text info at top
    txt = (
        f"VIIRS Edge Analysis Report\n"
        f"--------------------------\n"
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"Data Source: {os.path.basename(DATA_DIR)}\n"
        f"Scenes Processed: {count_processed}\n"
        f"Latitude Threshold: > {LATITUDE_THRESHOLD} N\n"
        f"Execution Time: {datetime.datetime.now() - start_time}\n"
    )
    plt.figtext(0.1, 0.8, txt, fontsize=12, family='monospace')

    # Polar Map at bottom
    if valid_lat_centers:
        ax = fig.add_axes([0.1, 0.05, 0.8, 0.7], projection='polar')
        
        # Convert Lat/Lon to polar coordinates
        # Theta = Longitude in Radians
        # R = Distance from pole (90 - Lat)
        theta = np.radians(valid_lon_centers)
        r = 90 - np.array(valid_lat_centers)
        
        ax.scatter(theta, r, c='blue', s=10, alpha=0.5)
        
        # Setup Arctic View
        ax.set_ylim(0, 90 - LATITUDE_THRESHOLD + 5) # Show from Pole down to threshold
        ax.set_yticks(np.arange(0, 90 - LATITUDE_THRESHOLD, 10))
        ax.set_yticklabels([]) # Hide radial labels
        ax.set_theta_zero_location('N') # 0 deg (Greenwich) at Top
        ax.set_theta_direction(-1)      # Clockwise
        ax.set_title(f"Geographic Coverage (Center Points)\n(N={len(valid_lat_centers)})")
    
    pdf.savefig(fig)
    plt.close()

    # --- PAGE 2: DEBUG MAP (If exists) ---
    for fig in debug_map_figs:
        pdf.savefig(fig)
        plt.close(fig)

    # --- PAGES 3+: HISTOGRAMS ---
    groups = {
        'Ice Neighbors':   ['iNw', 'i2Nw', 'i3Nw'],
        'Water Neighbors': ['wNi', 'w2Ni', 'w3Ni'],
        'Cloud Neighbors': ['iNc', 'i2Nc', 'wNc', 'w2Nc'],
        'Mixed Neighbors': ['Mixed']
    }

    # Order of plots: Channel, Diffs, Geometry, Textures
    plot_order = ['t11', 'diff1', 'diff2', 'diff3', 'sunz', 'satz', 
                  'r06_tex', 't11_tex', 't11t12_tex', 't37t12_tex']

    for acc_key in plot_order:
        accumulator = accs[acc_key]
        for group_name, keys in groups.items():
            accumulator.plot_group_to_pdf(pdf, keys, group_name)

print(f"Report saved successfully: {pdf_path}")