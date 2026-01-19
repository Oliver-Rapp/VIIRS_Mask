import datetime
import numpy as np
import multiprocessing as mp

# Import from the library package
from lib import config
from lib import data_io
from lib import core_logic
from lib import plotting

# Worker Function (Must be top-level for multiprocessing)
def process_scene_task(args):
    # Unpack arguments: timestamp, file paths, and boolean flag for plotting
    ts, files, do_plot = args 
    
    # 1. Load Data (IO Module)
    data = data_io.load_scene_data(files)
    if data is None: return None

    # 2. Classification & Masking (Logic Module)
    cls, cloud_clean = core_logic.create_classification(
        data['cma'], data['landuse'], data['lat'], config.LATITUDE_THRESHOLD
    )

    # 3. Check for sufficient Ice (Optimization)
    # Checks if we have enough ice pixels to bother processing stats
    if np.count_nonzero(cls == 2) < config.MIN_ICE_PIXELS:
        return None

    # --- DEBUG MAPPING ---
    # If this is one of the first N scenes, save a map image
    if do_plot:
        plotting.save_scene_map(cls, ts)

    # 4. Neighbors & Stats (Logic Module)
    nbs = core_logic.compute_neighbors(cls, cloud_clean)
    
    # 5. Local Histograms
    local_counts = {acc: {k: np.zeros(len(config.BINS[acc])-1) for k in config.KEYS} 
                    for acc in config.BINS.keys()}
    
    # Map data keys to bin keys
    var_map = {
        't11': 't11', 'diff1': 'diff1', 'diff2': 'diff2', 'diff3': 'diff3',
        'sunz': 'sunz', 'satz': 'satz',
        'r06_tex': 'r06_tex', 't11_tex': 't11_tex', 
        't11t12_tex': 't11t12_tex', 't37t12_tex': 't37t12_tex'
    }

    for key in config.KEYS: # Iterate mask types (iNw, Mixed...)
        if key not in nbs: continue
        mask = nbs[key]
        if not np.any(mask): continue
        
        for var_name, bin_name in var_map.items():
            vals = data[var_name][mask]
            # Handle masked arrays
            if np.ma.is_masked(vals): vals = vals.compressed()
            if len(vals) > 0:
                hist, _ = np.histogram(vals, bins=config.BINS[bin_name])
                local_counts[bin_name][key] += hist

    # 6. Mosaic Data (Subsampling)
    s = config.MOSAIC_STEP
    cls_sub = cls[::s, ::s].flatten()
    valid = cls_sub > 0
    mosaic = {
        'lat': data['lat'][::s, ::s].flatten()[valid],
        'lon': data['lon'][::s, ::s].flatten()[valid],
        'cls': cls_sub[valid]
    }

    return {'counts': local_counts, 'mosaic': mosaic, 'ts': ts}

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    
    print(f"--- VIIRS PRODUCTION PIPELINE ---")
    print(f"Directory: {config.DATA_DIR}")
    
    file_groups = data_io.get_file_groups(config.DATA_DIR)
    print(f"Found {len(file_groups)} items. Processing...")

    # --- PREPARE TASKS WITH PLOT FLAG ---
    # We create a list of arguments to pass to the workers.
    # The third argument is 'True' for the first N scenes, determining if they should output a .png
    tasks = []
    for i, (ts, files) in enumerate(file_groups.items()):
        should_plot = (i < config.NUM_DEBUG_MAPS)
        tasks.append((ts, files, should_plot))

    # Master Accumulator
    master_counts = {acc: {k: np.zeros(len(config.BINS[acc])-1) for k in config.KEYS} 
                     for acc in config.BINS.keys()}
    mosaic_data = {'lat': [], 'lon': [], 'cls': []}
    timestamps = []
    
    # Run Pool
    pool_size = config.NUM_WORKERS if config.NUM_WORKERS else mp.cpu_count()
    processed_count = 0
    print(f"Using {pool_size} CPU cores.")
    
    with mp.Pool(pool_size) as pool:
        for result in pool.imap_unordered(process_scene_task, tasks):
            if result is None: continue
            
            # Aggregate Histograms
            for acc, data in result['counts'].items():
                for key, hist in data.items():
                    master_counts[acc][key] += hist
            
            # Aggregate Mosaic
            mosaic_data['lat'].append(result['mosaic']['lat'])
            mosaic_data['lon'].append(result['mosaic']['lon'])
            mosaic_data['cls'].append(result['mosaic']['cls'])
            
            timestamps.append(result['ts'])
            processed_count += 1
            if processed_count % 10 == 0: print(f"Processed {processed_count}...")

    # Metadata
    timestamps.sort()
    meta = {
        'count': processed_count,
        'start': timestamps[0] if timestamps else 'N/A',
        'end': timestamps[-1] if timestamps else 'N/A',
        'duration': datetime.datetime.now() - start_time
    }

    # Generate Report
    plotting.generate_report(meta, master_counts, mosaic_data)
    print("Done.")