import os
import re
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    HAS_CARTOPY = False

# ================= CONFIGURATION =================

# Set to TRUE to reproduce the single-scene graphs exactly
VALIDATION_MODE = False

if VALIDATION_MODE:
    # Point this to the folder containing ONLY the single case files
    DATA_DIR = '/home/oliver/Documents/MET/VIIRS_Mask/data-case1'
    OUTPUT_DIR = '/home/oliver/Documents/MET/VIIRS_Mask/output_validation_exact_v2'
    PDF_FILENAME = 'Validation_Report_v2.pdf'
    
    USE_CROP = True
    CROP_X = (2005, 2286)
    CROP_Y = (610, 749)
    LATITUDE_THRESHOLD = -90.0 # Ignore lat check
    NUM_WORKERS = 1  
    MOSAIC_STEP = 1  
else:
    # Full day processing
    DATA_DIR = '/home/oliver/Documents/MET/VIIRS_Mask/20250415_noaa21-viirs-pps'
    OUTPUT_DIR = '/home/oliver/Documents/MET/VIIRS_Mask/output_day_stats_final'
    PDF_FILENAME = 'VIIRS_Daily_Report.pdf'
    
    USE_CROP = False
    CROP_X = None; CROP_Y = None
    LATITUDE_THRESHOLD = 60.0
    NUM_WORKERS = None 
    MOSAIC_STEP = 40

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ================= BINS (ADJUSTED FOR MATCHING) =================
# Bins updated to cover full range seen in original plots
BINS = {
    # Thermal: 255-275 focus, but covering tails
    't11': np.linspace(240, 280, 81),       
    
    # Differences: Matched to typical physical ranges
    'diff1': np.linspace(0, 3, 61),         # T11-T12
    'diff2': np.linspace(0, 12, 61),        # T3.7-T12
    'diff3': np.linspace(-3, 1.5, 91),      # T8.7-T12
    
    # Geometry
    'sunz': np.linspace(0, 90, 91),
    'satz': np.linspace(0, 70, 71),
    
    # Textures: Expanded ranges based on your screenshots
    'r06_tex': np.linspace(0, 40, 81),      # Expanded to 40 to capture tail
    't11_tex': np.linspace(0, 5, 51),       
    't11t12_tex': np.linspace(0, 0.5, 51),  
    't37t12_tex': np.linspace(0, 6, 61)     
}

# Strict Plotting Limits (Applied during PDF generation)
XLIMS = {
    't11': (255, 275),
    'diff1': (0, 2),
    'diff2': (0, 11),
    'diff3': (-2.5, 1),
    't11t12_tex': (0, 0.3),
    't37t12_tex': (0, 4)
    # r06_tex limit removed to let it show the full 0-40 range
}

KEYS = ['iNw', 'i2Nw', 'i3Nw', 'wNi', 'w2Ni', 'w3Ni', 'iNc', 'i2Nc', 'wNc', 'w2Nc', 'Mixed']

# ================= HELPER FUNCTIONS =================
def compute_all_neighbors(classification, cloud_bool):
    struct3 = ndimage.generate_binary_structure(2, 1)
    ice_bool = (classification == 2)
    water_bool = (classification == 1)

    # Ice Neighbors
    dilated_water = ndimage.binary_dilation(water_bool, structure=struct3)
    iNw = dilated_water & ice_bool & (~water_bool)
    i2Nw = ndimage.binary_dilation(iNw, structure=struct3) & ice_bool & (~iNw)
    i3Nw = ndimage.binary_dilation(i2Nw, structure=struct3) & ice_bool & (~iNw) & (~i2Nw)

    # Water Neighbors
    dilated_ice = ndimage.binary_dilation(ice_bool, structure=struct3)
    wNi = dilated_ice & water_bool & (~ice_bool)
    w2Ni = ndimage.binary_dilation(wNi, structure=struct3) & water_bool & (~wNi)
    w3Ni = ndimage.binary_dilation(w2Ni, structure=struct3) & water_bool & (~wNi) & (~w2Ni)

    # Cloud Neighbors
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
    
    if VALIDATION_MODE:
        # Validation Mode: 
        # We assume the directory contains exactly ONE scene's worth of files.
        # We create a single group 'Validation_Scene' and populate it.
        groups['Validation_Scene'] = {}
        for f in files:
            full_path = os.path.join(directory, f)
            if 'CMA' in f: groups['Validation_Scene']['cma'] = full_path
            elif 'physiography' in f: groups['Validation_Scene']['geo'] = full_path
            elif 'viirs' in f: groups['Validation_Scene']['l1b'] = full_path
            elif 'textures' in f: groups['Validation_Scene']['tex'] = full_path
    else:
        # Production Mode: Regex matching for timestamps
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
        
        # Robust lat reading
        lat = None
        for v in ['lat', 'latitude', 'nav_lat']:
            if v in ds_l1b.variables:
                lat = np.squeeze(ds_l1b.variables[v][:])
                break
        
        if lat is None: return None
        if not VALIDATION_MODE and np.max(lat) < LATITUDE_THRESHOLD: return None
        
        # Robust lon reading
        lon = None
        for v in ['lon', 'longitude', 'nav_lon']:
            if v in ds_l1b.variables:
                lon = np.squeeze(ds_l1b.variables[v][:])
                break
        if lon is None: lon = np.zeros_like(lat)

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
        
        r06_tex = ds_tex.variables['r06'][0,:,:].astype(float)
        t11_tex = ds_tex.variables['t11'][0,:,:].astype(float)
        t11t12_tex = ds_tex.variables['t11t12'][0,:,:].astype(float)
        t37t12_tex = ds_tex.variables['t37t12'][0,:,:].astype(float)

        # --- CROP LOGIC ---
        if USE_CROP:
            ys, ye = CROP_Y[0], CROP_Y[1] + 1
            xs, xe = CROP_X[0], CROP_X[1] + 1
            
            t11 = t11[ys:ye, xs:xe]; t12 = t12[ys:ye, xs:xe]
            t37 = t37[ys:ye, xs:xe]; t87 = t87[ys:ye, xs:xe]
            sunz = sunz[ys:ye, xs:xe]; satz = satz[ys:ye, xs:xe]
            cma = cma[ys:ye, xs:xe]; landuse = landuse[ys:ye, xs:xe]
            lat = lat[ys:ye, xs:xe]; lon = lon[ys:ye, xs:xe]
            r06_tex = r06_tex[ys:ye, xs:xe]; t11_tex = t11_tex[ys:ye, xs:xe]
            t11t12_tex = t11t12_tex[ys:ye, xs:xe]; t37t12_tex = t37t12_tex[ys:ye, xs:xe]

        data_map = {
            't11': t11, 'sunz': sunz, 'satz': satz,
            'diff1': t11-t12, 'diff2': t37-t12, 'diff3': t87-t12,
            'r06_tex': r06_tex, 't11_tex': t11_tex,
            't11t12_tex': t11t12_tex, 't37t12_tex': t37t12_tex
        }

        # --- MASKS ---
        if not USE_CROP: lat_mask = (lat >= LATITUDE_THRESHOLD)
        else: lat_mask = np.ones_like(lat, dtype=bool)

        cloud_clean = ((cma == 1) | (cma == 2))
        cloud_clean[landuse != 16] = False 
        
        classification = np.zeros_like(cma, dtype=np.uint8)
        classification[(landuse == 16) & (cma == 0)] = 1 
        classification[(cma == 3)] = 2                   
        classification[cloud_clean] = 3                  
        classification[(landuse != 16)] = 4               
        classification[~lat_mask] = 0

        # In validation, we assume the user picked a valid scene, so no minimum ice check.
        if not VALIDATION_MODE and np.count_nonzero(classification == 2) < 200: 
            return None

        # --- MOSAIC ---
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

        # --- STATS ---
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

        return {'counts': local_counts, 'mosaic': mosaic_data, 'ts': ts}

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
    
    print(f"--- VIIRS ANALYSIS ---")
    print(f"Mode: {'VALIDATION (Fixed Crop)' if VALIDATION_MODE else 'PRODUCTION'}")
    
    file_groups = get_file_groups(DATA_DIR)
    print(f"Found {len(file_groups)} items/scenes. Processing...")

    master_counts = {acc: {k: np.zeros(len(BINS[acc])-1) for k in KEYS} for acc in BINS.keys()}
    mosaic_lats = []
    mosaic_lons = []
    mosaic_clss = []
    processed_timestamps = []
    
    count_processed = 0
    pool_size = NUM_WORKERS if NUM_WORKERS else mp.cpu_count()

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
            
            processed_timestamps.append(result['ts'])
            count_processed += 1
            if count_processed % 10 == 0: print(f"Processed {count_processed}...")

    end_time = datetime.datetime.now()
    processed_timestamps.sort()
    ts_start = processed_timestamps[0] if processed_timestamps else "N/A"
    ts_end = processed_timestamps[-1] if processed_timestamps else "N/A"

    print(f"Generating PDF: {OUTPUT_DIR}/{PDF_FILENAME}")
    pdf_path = os.path.join(OUTPUT_DIR, PDF_FILENAME)
    
    with PdfPages(pdf_path) as pdf:
        # --- METADATA & MAP ---
        fig = plt.figure(figsize=(11, 12))
        meta = (
            f"VIIRS EDGE ANALYSIS\n"
            f"===================\n"
            f"Date: {start_time.strftime('%Y-%m-%d %H:%M')}\n"
            f"Mode: {'VALIDATION' if VALIDATION_MODE else 'FULL DAY'}\n"
            f"Dir:  {os.path.basename(DATA_DIR)}\n"
            f"Count: {count_processed}\n"
            f"Start: {ts_start}\n"
            f"End:   {ts_end}\n"
            f"Lat Limit: {LATITUDE_THRESHOLD}\n"
            f"Crop: {USE_CROP} {CROP_X if USE_CROP else ''}\n"
        )
        plt.figtext(0.05, 0.7, meta, fontsize=11, family='monospace', va='top')

        if HAS_CARTOPY and mosaic_lats:
            ax = fig.add_axes([0.35, 0.05, 0.6, 0.6], projection=ccrs.NorthPolarStereo())
            ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor='darkgrey')
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            
            all_lats = np.concatenate(mosaic_lats)
            all_lons = np.concatenate(mosaic_lons)
            all_clss = np.concatenate(mosaic_clss)
            cmap_mosaic = ListedColormap(['blue', 'white', 'silver', 'green'])
            
            ax.scatter(all_lons, all_lats, c=all_clss, cmap=cmap_mosaic, 
                       vmin=1, vmax=4, s=0.5 if VALIDATION_MODE else 0.1, 
                       transform=ccrs.Geodetic())
            ax.set_title(f"Data Mosaic (N={count_processed})")
        else:
            plt.figtext(0.6, 0.4, "Map Unavailable", ha='center')
        
        pdf.savefig(fig); plt.close()

        # --- PLOTS ---
        groups = {
            'Ice Neighbors':   ['iNw', 'i2Nw', 'i3Nw'],
            'Water Neighbors': ['wNi', 'w2Ni', 'w3Ni'],
            'Cloud Neighbors': ['iNc', 'i2Nc', 'wNc', 'w2Nc'],
            'Mixed Neighbors': ['Mixed']
        }
        
        plot_order = [
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

        for acc_name, title in plot_order:
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
                
                plt.xlabel(title)
                plt.ylabel("Density")
                plt.title(f"{title} ({group_name})")
                
                if acc_name in XLIMS: plt.xlim(XLIMS[acc_name])
                
                if has_data: plt.legend()
                else: plt.text(0.5, 0.5, "No Data", ha='center', transform=plt.gca().transAxes)
                
                plt.grid(True, linestyle="--", alpha=0.5)
                plt.tight_layout()
                pdf.savefig()
                plt.close()

    print("Success.")