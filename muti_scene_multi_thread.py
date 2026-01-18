import os
import re
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # <--- THIS WAS MISSING
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from scipy import ndimage
from netCDF4 import Dataset
import multiprocessing as mp

# --- MAP IMPORTS ---
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    print("WARNING: Cartopy not installed. Map will be skipped.")
    HAS_CARTOPY = False

# ================= CONFIGURATION =================
DATA_DIR = '/home/oliver/Documents/MET/VIIRS_Mask/20250415_noaa21-viirs-pps'
OUTPUT_DIR = '/home/oliver/Documents/MET/VIIRS_Mask/output_day_stats_full'
PDF_FILENAME = 'VIIRS_Analysis_Report.pdf'
LATITUDE_THRESHOLD = 60.0
NUM_WORKERS = None 

# MOSAIC SETTINGS
# Step size for subsampling. 
# 40 = faster, coarse dots. 20 = slower, fine dots.
MOSAIC_STEP = 40 

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ================= BINS & DEFINITIONS =================
BINS = {
    't11': np.linspace(230, 290, 121),
    'diff1': np.linspace(-1, 5, 121),
    'diff2': np.linspace(-5, 15, 101),
    'diff3': np.linspace(-4, 4, 81),
    'sunz': np.linspace(0, 90, 91),
    'satz': np.linspace(0, 70, 71),
    'r06_tex': np.linspace(0, 10, 101),
    't11_tex': np.linspace(0, 5, 101),
    't11t12_tex': np.linspace(0, 1, 101),
    't37t12_tex': np.linspace(0, 8, 101)
}

KEYS = ['iNw', 'i2Nw', 'i3Nw', 'wNi', 'w2Ni', 'w3Ni', 'iNc', 'i2Nc', 'wNc', 'w2Nc', 'Mixed']

# ================= HELPER FUNCTIONS =================
def compute_all_neighbors(classification, cloud_bool):
    struct3 = ndimage.generate_binary_structure(2, 1)
    ice_bool = (classification == 2)
    water_bool = (classification == 1)

    dilated_water = ndimage.binary_dilation(water_bool, structure=struct3)
    iNw = dilated_water & ice_bool & (~water_bool)
    i2Nw = ndimage.binary_dilation(iNw, structure=struct3) & ice_bool & (~iNw)
    i3Nw = ndimage.binary_dilation(i2Nw, structure=struct3) & ice_bool & (~iNw) & (~i2Nw)

    dilated_ice = ndimage.binary_dilation(ice_bool, structure=struct3)
    wNi = dilated_ice & water_bool & (~ice_bool)
    w2Ni = ndimage.binary_dilation(wNi, structure=struct3) & water_bool & (~wNi)
    w3Ni = ndimage.binary_dilation(w2Ni, structure=struct3) & water_bool & (~wNi) & (~w2Ni)

    iNc = ndimage.binary_dilation(cloud_bool, structure=struct3) & ice_bool
    i2Nc = ndimage.binary_dilation(iNc, structure=struct3) & ice_bool & (~iNc)
    wNc = ndimage.binary_dilation(cloud_bool, structure=struct3) & water_bool
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

# ================= WORKER FUNCTION =================
def process_scene(args):
    ts, files = args
    if not all(k in files for k in ['cma', 'geo', 'l1b', 'tex']): return None 

    ds_l1b = None; ds_cma = None; ds_geo = None; ds_tex = None
    
    local_counts = {acc: {k: np.zeros(len(BINS[acc])-1) for k in KEYS} for acc in BINS.keys()}
    mosaic_data = None

    try:
        ds_l1b = Dataset(files['l1b'], 'r')
        if 'lat' not in ds_l1b.variables: return None
        
        lat = np.squeeze(ds_l1b.variables['lat'][:])
        if np.max(lat) < LATITUDE_THRESHOLD: return None
        
        lon_var = next((k for k in ['lon', 'longitude', 'nav_lon'] if k in ds_l1b.variables), None)
        if lon_var is None: return None
        lon = np.squeeze(ds_l1b.variables[lon_var][:])

        # Load satellite data
        t11 = ds_l1b.variables['image3'][0, :, :]
        t12 = ds_l1b.variables['image4'][0, :, :]
        t37 = ds_l1b.variables['image5'][0, :, :]
        t87 = ds_l1b.variables['image7'][0, :, :]
        sunz = ds_l1b.variables['sunzenith'][0, :, :]
        satz = ds_l1b.variables['satzenith'][0, :, :]

        ds_cma = Dataset(files['cma'], 'r')
        ds_geo = Dataset(files['geo'], 'r')
        ds_tex = Dataset(files['tex'], 'r')

        cma = ds_cma.variables['cma_extended'][0, :, :]
        landuse = ds_geo.variables['landuse'][0, :, :]
        
        data_map = {
            't11': t11, 'sunz': sunz, 'satz': satz,
            'diff1': t11-t12, 'diff2': t37-t12, 'diff3': t87-t12,
            'r06_tex': ds_tex.variables['r06'][0,:,:].astype(float),
            't11_tex': ds_tex.variables['t11'][0,:,:].astype(float),
            't11t12_tex': ds_tex.variables['t11t12'][0,:,:].astype(float),
            't37t12_tex': ds_tex.variables['t37t12'][0,:,:].astype(float)
        }

        lat_mask = (lat >= LATITUDE_THRESHOLD)
        cloud_clean = ((cma == 1) | (cma == 2))
        cloud_clean[landuse != 16] = False
        
        classification = np.zeros_like(cma, dtype=np.uint8)
        classification[(landuse == 16) & (cma == 0)] = 1 
        classification[(cma == 3)] = 2                   
        classification[cloud_clean] = 3                  
        classification[(landuse != 16)] = 4               
        classification[~lat_mask] = 0

        if np.count_nonzero(classification == 2) < 200: return None

        # --- EXTRACT MOSAIC POINTS ---
        s = MOSAIC_STEP
        cls_sub = classification[::s, ::s].flatten()
        lat_sub = lat[::s, ::s].flatten()
        lon_sub = lon[::s, ::s].flatten()
        
        valid_idx = cls_sub > 0
        mosaic_data = {
            'lat': lat_sub[valid_idx],
            'lon': lon_sub[valid_idx],
            'cls': cls_sub[valid_idx]
        }

        # --- CALCULATE STATS ---
        nbs = compute_all_neighbors(classification, cloud_clean)
        
        neighbor_sum = np.zeros_like(classification, dtype=int)
        for k in ['iNw','i2Nw','wNi','w2Ni','iNc','i2Nc','wNc','w2Nc']:
            if k in nbs: neighbor_sum += nbs[k].astype(int)
        nbs['Mixed'] = (neighbor_sum >= 2)

        for key in KEYS:
            if key not in nbs: continue
            mask = nbs[key]
            if not np.any(mask): continue
            
            for d_name, d_val in data_map.items():
                vals = d_val[mask]
                if np.ma.is_masked(vals): vals = vals.compressed()
                if len(vals) > 0:
                    hist, _ = np.histogram(vals, bins=BINS[d_name])
                    local_counts[d_name][key] += hist

        return {'counts': local_counts, 'mosaic': mosaic_data}

    except Exception:
        return None
    finally:
        for d in [ds_l1b, ds_cma, ds_geo, ds_tex]:
            try: 
                if d is not None: d.close()
            except: pass

# ================= MAIN EXECUTION =================
if __name__ == '__main__':
    start_time = datetime.datetime.now()
    
    file_groups = get_file_groups(DATA_DIR)
    print(f"Found {len(file_groups)} scenes. Starting parallel processing...")

    master_counts = {acc: {k: np.zeros(len(BINS[acc])-1) for k in KEYS} for acc in BINS.keys()}
    
    mosaic_lats = []
    mosaic_lons = []
    mosaic_clss = []
    
    count_processed = 0

    pool_size = NUM_WORKERS if NUM_WORKERS else mp.cpu_count()
    print(f"Using {pool_size} CPU cores...")

    with mp.Pool(pool_size) as pool:
        for result in pool.imap_unordered(process_scene, file_groups.items()):
            if result is None: continue
            
            for acc_name, acc_data in result['counts'].items():
                for key, hist in acc_data.items():
                    master_counts[acc_name][key] += hist
            
            if result['mosaic']:
                mosaic_lats.append(result['mosaic']['lat'])
                mosaic_lons.append(result['mosaic']['lon'])
                mosaic_clss.append(result['mosaic']['cls'])
            
            count_processed += 1
            if count_processed % 10 == 0:
                print(f"Processed {count_processed} valid scenes...")

    print(f"Processing done. Generating PDF report...")

    pdf_path = os.path.join(OUTPUT_DIR, PDF_FILENAME)
    
    with PdfPages(pdf_path) as pdf:
        # --- PAGE 1: MOSAIC MAP ---
        fig = plt.figure(figsize=(12, 12))
        
        if HAS_CARTOPY and mosaic_lats:
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
            ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())
            
            ax.add_feature(cfeature.LAND, facecolor='darkgrey', zorder=1)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
            ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--', zorder=3)

            all_lats = np.concatenate(mosaic_lats)
            all_lons = np.concatenate(mosaic_lons)
            all_clss = np.concatenate(mosaic_clss)

            # 1=Water(Blue), 2=Ice(White), 3=Cloud(Silver), 4=Land(Green)
            cmap_mosaic = ListedColormap(['blue', 'white', 'silver', 'green'])
            
            sc = ax.scatter(all_lons, all_lats, c=all_clss, cmap=cmap_mosaic, 
                            vmin=1, vmax=4, s=0.5, alpha=1.0, transform=ccrs.Geodetic(), zorder=1.5)
            
            patches = [
                mpatches.Patch(color='blue', label='Water'),
                mpatches.Patch(color='white', label='Ice'),
                mpatches.Patch(color='silver', label='Cloud'),
                mpatches.Patch(color='green', label='Land')
            ]
            ax.legend(handles=patches, loc='lower left', title="Classification")
            
            ax.set_title(f"VIIRS Daily Mosaic (Subsampled)\nLatitude > {LATITUDE_THRESHOLD}N | N={count_processed} Scenes", fontsize=14)
        else:
            plt.axis('off')
            plt.text(0.5, 0.5, "Map Unavailable", ha='center')

        txt = (
            f"Execution Time: {datetime.datetime.now() - start_time}\n"
            f"Data Source: {os.path.basename(DATA_DIR)}"
        )
        plt.figtext(0.1, 0.05, txt, fontsize=10, family='monospace')
        
        pdf.savefig(fig)
        plt.close()

        # --- PAGES 2+: HISTOGRAMS ---
        groups = {
            'Ice Neighbors':   ['iNw', 'i2Nw', 'i3Nw'],
            'Water Neighbors': ['wNi', 'w2Ni', 'w3Ni'],
            'Cloud Neighbors': ['iNc', 'i2Nc', 'wNc', 'w2Nc'],
            'Mixed Neighbors': ['Mixed']
        }
        
        plot_order = ['t11', 'diff1', 'diff2', 'diff3', 'sunz', 'satz', 
                      'r06_tex', 't11_tex', 't11t12_tex', 't37t12_tex']
        
        titles = {
            't11': 'T11 (K)', 'diff1': 'T11-T12 (K)', 'diff2': 'T3.7-T12 (K)', 'diff3': 'T8.7-T12 (K)',
            'sunz': 'Sun Zenith', 'satz': 'Sat Zenith', 'r06_tex': 'r06-tex', 
            't11_tex': 't11-tex', 't11t12_tex': 't11t12-tex', 't37t12_tex': 't37t12-tex'
        }
        xlims = {
            't11': (240, 280), 'diff1': (0, 2.5), 'diff2': (0, 11), 'diff3': (-2.5, 1),
            't11t12_tex': (0, 0.3), 't37t12_tex': (0, 6)
        }

        for acc_name in plot_order:
            bins = BINS[acc_name]
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            for group_name, keys in groups.items():
                plt.figure(figsize=(10, 6))
                has_data = False
                
                for key in keys:
                    counts = master_counts[acc_name][key]
                    total = np.sum(counts)
                    if total > 0:
                        has_data = True
                        plt.plot(bin_centers, counts/total, label=f"{key} (n={int(total)})", linewidth=2)
                
                plt.xlabel(titles[acc_name])
                plt.ylabel("Density")
                plt.title(f"{titles[acc_name]} ({group_name})")
                if acc_name in xlims: plt.xlim(xlims[acc_name])
                if has_data: plt.legend()
                else: plt.text(0.5, 0.5, "No Data", ha='center', transform=plt.gca().transAxes)
                plt.grid(True, linestyle="--", alpha=0.5)
                plt.tight_layout()
                pdf.savefig()
                plt.close()

    print(f"Report saved: {pdf_path}")