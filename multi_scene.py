import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg') # Server-safe backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from scipy import ndimage
from netCDF4 import Dataset

# ================= CONFIGURATION =================
DATA_DIR = '/home/oliver/Documents/MET/VIIRS_Mask/20250415_noaa21-viirs-pps'
OUTPUT_DIR = '/home/oliver/Documents/MET/VIIRS_Mask/output_day_stats_full'
LATITUDE_THRESHOLD = 60.0
SAVE_DEBUG_MAP = True  # Saves the map for the FIRST valid scene only

# Setup Output
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ================= COLORS & STYLES (From Original) =================
colors = [
    'black', 'blue', 'white', 'gray', 'green', # 0-4
    'red', 'orange', 'gold',                   # 5-7 (Ice NBs)
    'cyan', 'navy', 'purple'                   # 8-10 (Water NBs)
]
labels = ['No data', 'Water', 'Ice', 'Clouds', 'Land', 'InW', 'I2nW', 'I3nW', 'WnI', 'W2nI', 'W3nI']
cmap = ListedColormap(colors)

# ================= ACCUMULATOR CLASS =================
class HistogramAccumulator:
    def __init__(self, name, bins, xlim=None):
        self.name = name
        self.bins = bins
        self.xlim = xlim
        # Store counts for EVERY distinct mask type defined in original script
        self.keys = [
            'iNw', 'i2Nw', 'i3Nw',
            'wNi', 'w2Ni', 'w3Ni',
            'iNc', 'i2Nc', 'wNc', 'w2Nc',
            'Mixed'
        ]
        self.counts = {k: np.zeros(len(bins)-1) for k in self.keys}

    def add(self, key, data_values):
        if len(data_values) == 0: return
        # Handle masked arrays (NetCDF often returns them)
        if np.ma.is_masked(data_values):
            data_values = data_values.compressed()
        
        hist, _ = np.histogram(data_values, bins=self.bins)
        self.counts[key] += hist

    def plot_group(self, group_keys, group_name, filename):
        """Plots a specific subset of keys (e.g., only Ice neighbors)"""
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
            plt.text(0.5, 0.5, "No Data", ha='center', transform=plt.gca().transAxes)
            
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        plt.close()

# ================= HELPER FUNCTIONS =================
def compute_all_neighbors(classification, cloud_bool):
    """Exact logic from original script."""
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

def plot_case_map(classification, neighbor_masks, cloud_neighbor_masks, mixed_mask, title, filename):
    """Replicates the original visualization style."""
    fig, ax = plt.subplots(figsize=(10, 8))
    # Base map
    ax.imshow(classification, interpolation='none', cmap=cmap, vmin=0, vmax=10)
    ax.set_title(title)
    ax.axis('off')

    # Draw Standard Neighbors
    color_map = {
        'iNw': colors[5], 'i2Nw': colors[6], 'i3Nw': colors[7],
        'wNi': colors[8], 'w2Ni': colors[9], 'w3Ni': colors[10]
    }
    for name, mask in neighbor_masks.items():
        if name in color_map:
            # Using spy for faster plotting than adding individual rectangles
            # Or masking the image overlay. Since we have classification array, 
            # we can just update the classification array for the plot if we wanted,
            # but here we follow the rectangle overlay logic (approximated for speed)
            y, x = np.where(mask)
            ax.scatter(x, y, c=color_map[name], s=1, marker='s', edgecolors='none')

    # Draw Cloud Neighbors
    for name, mask in cloud_neighbor_masks.items():
        c = 'lightslategray' if '2' in name else 'lightgray'
        y, x = np.where(mask)
        ax.scatter(x, y, c=c, s=1, marker='s', edgecolors='none')

    # Draw Mixed
    if mixed_mask is not None:
        y, x = np.where(mixed_mask)
        ax.scatter(x, y, c='orchid', s=1, marker='s', edgecolors='none')

    # Simple Legend
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors[1:5], labels[1:5])]
    patches.append(mpatches.Patch(color='orchid', label='Mixed'))
    ax.legend(handles=patches, loc='lower right', fontsize='small')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

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

# ================= INITIALIZE ACCUMULATORS =================
# Defining all histogram objects requested in the original script
# Note: Bins slightly widened to accommodate full-day variability
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

# ================= PROCESSING LOOP =================
file_groups = get_file_groups(DATA_DIR)
print(f"Found {len(file_groups)} scenes.")
count_processed = 0

for ts, files in file_groups.items():
    if not all(k in files for k in ['cma', 'geo', 'l1b', 'tex']): continue

    ds_l1b = None; ds_cma = None; ds_geo = None; ds_tex = None

    try:
        # 1. Latitude Check (from VIIRS file)
        ds_l1b = Dataset(files['l1b'], 'r')
        if 'lat' not in ds_l1b.variables: continue
        lat = np.squeeze(ds_l1b.variables['lat'][:])
        if np.max(lat) < LATITUDE_THRESHOLD: continue

        # 2. Load Satellite Data
        # image3=T11, image4=T12, image5=T3.7, image7=T8.7
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
        
        # Textures
        tex_data = {
            'r06_tex': ds_tex.variables['r06'][0,:,:].astype(float),
            't11_tex': ds_tex.variables['t11'][0,:,:].astype(float),
            't11t12_tex': ds_tex.variables['t11t12'][0,:,:].astype(float),
            't37t12_tex': ds_tex.variables['t37t12'][0,:,:].astype(float)
        }

        # 4. Generate Masks (Exact Original Logic)
        lat_mask = (lat >= LATITUDE_THRESHOLD)
        is_land = (landuse != 16)
        
        cloud_raw = ((cma == 1) | (cma == 2))
        cloud_clean = cloud_raw.copy()
        cloud_clean[is_land] = False
        
        classification = np.zeros_like(cma, dtype=np.uint8)
        classification[(landuse == 16) & (cma == 0)] = 1 # Water
        classification[(cma == 3)] = 2                   # Ice
        classification[cloud_clean] = 3                  # Cloud
        classification[~lat_mask] = 0

        if np.count_nonzero(classification == 2) < 200: continue

        # 5. Compute Neighbors
        nbs = compute_all_neighbors(classification, cloud_clean)
        
        # 6. Compute Mixed (Exact sum logic)
        neighbor_sum = np.zeros_like(classification, dtype=int)
        for k in ['iNw','i2Nw','wNi','w2Ni','iNc','i2Nc','wNc','w2Nc']:
            if k in nbs: neighbor_sum += nbs[k].astype(int)
        nbs['Mixed'] = (neighbor_sum >= 2)

        # 7. Accumulate Data
        # Calc differences
        diffs = {
            'diff1': t11 - t12,
            'diff2': t37 - t12,
            'diff3': t87 - t12
        }

        # Loop over every neighbor type (iNw, wNi, Mixed, etc)
        for key in accs['t11'].keys: # Iterate keys defined in Accumulator
            if key not in nbs: continue
            mask = nbs[key]
            if not np.any(mask): continue

            # Add to all histograms
            accs['t11'].add(key, t11[mask])
            for dname, dval in diffs.items():
                accs[dname].add(key, dval[mask])
            accs['sunz'].add(key, sunz[mask])
            accs['satz'].add(key, satz[mask])
            for tname, tval in tex_data.items():
                accs[tname].add(key, tval[mask])

        # 8. Plot Debug Map (First valid scene)
        if SAVE_DEBUG_MAP and count_processed == 0:
            print(f"Generating Debug Map for {ts}...")
            # We filter nbs for the specific plot types used in original script
            i_w_nbs = {k: nbs[k] for k in ['iNw','i2Nw','i3Nw','wNi','w2Ni','w3Ni']}
            c_nbs   = {k: nbs[k] for k in ['iNc','i2Nc','wNc','w2Nc']}
            
            # Map 1: Ice/Water Neighbors
            plot_case_map(classification, i_w_nbs, {}, None, 
                          f"Ice/Water Neighbors ({ts})", "debug_map_ice_water.png")
            # Map 2: Cloud Neighbors + Mixed
            plot_case_map(classification, {}, c_nbs, nbs['Mixed'], 
                          f"Cloud/Mixed Neighbors ({ts})", "debug_map_cloud_mixed.png")

        count_processed += 1
        if count_processed % 10 == 0: print(f"Processed {count_processed}...")

    except Exception as e:
        print(f"Error {ts}: {e}")
    finally:
        for d in [ds_l1b, ds_cma, ds_geo, ds_tex]:
            try: 
                if d: d.close()
            except: pass

# ================= FINAL PLOTTING =================
print("Generating Final Histograms...")

# Define the Groups (Exact matches to original script)
groups = {
    'Cloud Neighbors': ['iNc', 'i2Nc', 'wNc', 'w2Nc'],
    'Ice Neighbors':   ['iNw', 'i2Nw', 'i3Nw'],
    'Water Neighbors': ['wNi', 'w2Ni', 'w3Ni'],
    'Mixed Neighbors': ['Mixed']
}

# Iterate over every accumulator (t11, diff1, textures...)
for acc_key, accumulator in accs.items():
    # Iterate over every group (Ice, Water, Cloud...)
    for group_name, keys in groups.items():
        # Construct filename like: hist_t11_Ice_Neighbors.png
        fname = f"hist_{acc_key}_{group_name.replace(' ','_')}.png"
        accumulator.plot_group(keys, group_name, fname)

print(f"Done. Processed {count_processed} scenes. Check {OUTPUT_DIR}")