import datetime
import numpy as np
import multiprocessing as mp
from lib import config, data_io, core_logic, plotting

def process_single_scene(args):
    ts, files, save_debug = args
    
    # 1. Load Data
    data = data_io.load_scene_data(files)
    if data is None: return None

    # 2. Classification
    cls, cloud_clean = core_logic.create_classification(
        data['cma'], data['landuse'], data['lat'], data['sunz']
    )

    # 3. Ice Check
    ice_count = np.count_nonzero(cls == 2)
    # DEBUG: print(f"{ts}: Ice count {ice_count}") 
    if ice_count < config.MIN_ICE_PIXELS: return None

    # --- CALCULATE COVERAGE ---
    valid_mask = (cls > 0)
    
    # Compute 2D histogram for this scene
    scene_coverage, _, _ = np.histogram2d(
        data['lon'][valid_mask][::10], 
        data['lat'][valid_mask][::10], 
        bins=[config.COVERAGE_LON_BINS, config.COVERAGE_LAT_BINS]
    )
    scene_coverage = (scene_coverage > 0).astype(int)

    # 4. Neighbors
    nbs = core_logic.compute_neighbors(cls, cloud_clean)
    
    # 5. Plotting
    if save_debug:
        plotting.generate_debug_suite(cls, nbs, ts)

    # 6. Histograms
    local_counts = {var: {k: np.zeros(len(config.BINS[var])-1) for k in config.KEYS} 
                    for var in config.BINS.keys()}
    
    for key in config.KEYS:
        if key not in nbs: continue
        mask = nbs[key]
        if not np.any(mask): continue
        
        for var_name in config.BINS.keys():
            if var_name not in data: continue
            
            vals = data[var_name][mask]
            
            # Handle masked arrays and NaNs
            if np.ma.is_masked(vals): vals = vals.compressed()
            vals = vals[~np.isnan(vals)]
            
            if len(vals) > 0:
                hist, _ = np.histogram(vals, bins=config.BINS[var_name])
                local_counts[var_name][key] += hist

    # --- FIX: INCLUDE 'ts' IN RETURN ---
    return {
        'ts': ts, 
        'counts': local_counts, 
        'coverage': scene_coverage
    }

def main():
    start_time = datetime.datetime.now()
    print("--- VIIRS PROCESSOR START ---")
    
    file_groups = data_io.get_file_groups(config.DATA_DIR)
    all_timestamps = sorted(list(file_groups.keys()))
    
    if config.Example_Run:
        all_timestamps = all_timestamps[:10]

    print(f"Found {len(all_timestamps)} scenes to process.")

    tasks = []
    for i, ts in enumerate(all_timestamps):
        do_debug = (i < config.NUM_DEBUG_MAPS)
        tasks.append((ts, file_groups[ts], do_debug))

    master_counts = {var: {k: np.zeros(len(config.BINS[var])-1) for k in config.KEYS} 
                     for var in config.BINS.keys()}
    
    # Grid for coverage map
    master_coverage = np.zeros((len(config.COVERAGE_LON_BINS)-1, len(config.COVERAGE_LAT_BINS)-1))

    processed_count = 0
    processed_timestamps = [] # NEW: Track list of successfully processed times
    
    with mp.Pool(config.NUM_WORKERS) as pool:
        for result in pool.imap_unordered(process_single_scene, tasks):
            if result is None: continue
            
            processed_count += 1
            processed_timestamps.append(result['ts']) # Store timestamp
            
            for var, key_dict in result['counts'].items():
                for key, hist in key_dict.items():
                    master_counts[var][key] += hist
            
            master_coverage += result['coverage']
            
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} scenes...")

    # Sort timestamps to find start/end
    processed_timestamps.sort()
    
    meta = {
        'count': processed_count,
        'duration': datetime.datetime.now() - start_time,
        'coverage_grid': master_coverage,
        # NEW: Add time span info
        'time_start': processed_timestamps[0] if processed_timestamps else "N/A",
        'time_end': processed_timestamps[-1] if processed_timestamps else "N/A"
    }
    
    if processed_count > 0:
        plotting.generate_pdf_report(meta, master_counts)
        print("Done. PDF generated.")
    else:
        print("No scenes processed (check thresholds/crop).")

if __name__ == '__main__':
    main()