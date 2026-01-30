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
    if ice_count < config.MIN_ICE_PIXELS: return None

    # --- CALCULATE COVERAGE ---
    valid_mask = (cls > 0)
    
    scene_coverage, _, _ = np.histogram2d(
        data['lon'][valid_mask][::10], 
        data['lat'][valid_mask][::10], 
        bins=[config.COVERAGE_LON_BINS, config.COVERAGE_LAT_BINS]
    )
    scene_coverage = (scene_coverage > 0).astype(int)

    # 4. Neighbors
    nbs = core_logic.compute_neighbors(cls, cloud_clean)
    
    # --- MIZ MASK CALCULATION ---
    # We calculate this here so we can use it for both Histograms AND Plotting
    miz_mask = None
    if config.ENABLE_MIZ_HISTOGRAMS:
        t_min, t_max = config.MIZ_T11_RANGE
        # Load T11
        t11 = data['t11']
        
        # Handle Masked Array if present to avoid warnings
        if np.ma.is_masked(t11):
            t11 = t11.filled(np.nan)
            
        # Create boolean mask
        with np.errstate(invalid='ignore'): # Silence NaN comparison warnings
            miz_mask = (t11 >= t_min) & (t11 <= t_max)

    # 5. Plotting
    if save_debug:
        # Pass the miz_mask to the plotter
        plotting.generate_debug_suite(cls, nbs, ts, miz_mask)

    # 6. Histograms
    local_counts = {var: {k: np.zeros(len(config.BINS[var])-1) for k in config.KEYS} 
                    for var in config.BINS.keys()}
    
    local_counts_miz = None
    if config.ENABLE_MIZ_HISTOGRAMS:
        local_counts_miz = {var: {k: np.zeros(len(config.BINS[var])-1) for k in config.KEYS} 
                            for var in config.BINS.keys()}
    
    for key in config.KEYS:
        if key not in nbs: continue
        mask = nbs[key]
        if not np.any(mask): continue
        
        # Calculate intersection of Class Mask and MIZ Mask
        miz_indices = None
        if miz_mask is not None:
            miz_indices = mask & miz_mask

        for var_name in config.BINS.keys():
            if var_name not in data: continue
            
            vals = data[var_name][mask]
            
            # --- Standard Histogram ---
            valid_vals = vals
            if np.ma.is_masked(valid_vals): valid_vals = valid_vals.compressed()
            valid_vals = valid_vals[~np.isnan(valid_vals)]
            
            if len(valid_vals) > 0:
                hist, _ = np.histogram(valid_vals, bins=config.BINS[var_name])
                local_counts[var_name][key] += hist

            # --- MIZ Subset Histogram ---
            if config.ENABLE_MIZ_HISTOGRAMS and miz_indices is not None:
                # We need values where BOTH (Class Mask) AND (MIZ Mask) are true
                # Since we are iterating over 'vals' which is data[mask],
                # we need to subset 'vals' using the MIZ condition relative to those pixels.
                
                # However, it is safer/cleaner to just index the original data 
                # using the combined boolean mask (miz_indices)
                vals_miz = data[var_name][miz_indices]
                
                if np.ma.is_masked(vals_miz): vals_miz = vals_miz.compressed()
                vals_miz = vals_miz[~np.isnan(vals_miz)]
                
                if len(vals_miz) > 0:
                    hist_miz, _ = np.histogram(vals_miz, bins=config.BINS[var_name])
                    local_counts_miz[var_name][key] += hist_miz

    return {
        'ts': ts, 
        'counts': local_counts, 
        'counts_miz': local_counts_miz,
        'coverage': scene_coverage
    }

def main():
    start_time = datetime.datetime.now()
    print("--- VIIRS PROCESSOR START ---")
    
    file_groups = data_io.get_file_groups(config.DATA_DIR)
    all_timestamps = sorted(list(file_groups.keys()))
    
    print(f"Found {len(all_timestamps)} scenes to process.")

    tasks = []
    for i, ts in enumerate(all_timestamps):
        do_debug = (i < config.NUM_DEBUG_MAPS)
        tasks.append((ts, file_groups[ts], do_debug))

    master_counts = {var: {k: np.zeros(len(config.BINS[var])-1) for k in config.KEYS} 
                     for var in config.BINS.keys()}
    
    master_counts_miz = None
    if config.ENABLE_MIZ_HISTOGRAMS:
        master_counts_miz = {var: {k: np.zeros(len(config.BINS[var])-1) for k in config.KEYS} 
                             for var in config.BINS.keys()}
    
    master_coverage = np.zeros((len(config.COVERAGE_LON_BINS)-1, len(config.COVERAGE_LAT_BINS)-1))

    processed_count = 0
    processed_timestamps = [] 
    
    with mp.Pool(config.NUM_WORKERS) as pool:
        for result in pool.imap_unordered(process_single_scene, tasks):
            if result is None: continue
            
            processed_count += 1
            processed_timestamps.append(result['ts'])
            
            for var, key_dict in result['counts'].items():
                for key, hist in key_dict.items():
                    master_counts[var][key] += hist
            
            if config.ENABLE_MIZ_HISTOGRAMS and result['counts_miz']:
                for var, key_dict in result['counts_miz'].items():
                    for key, hist in key_dict.items():
                        master_counts_miz[var][key] += hist
            
            master_coverage += result['coverage']
            
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} scenes...")

    processed_timestamps.sort()
    
    meta = {
        'count': processed_count,
        'duration': datetime.datetime.now() - start_time,
        'coverage_grid': master_coverage,
        'time_start': processed_timestamps[0] if processed_timestamps else "N/A",
        'time_end': processed_timestamps[-1] if processed_timestamps else "N/A"
    }
    
    if processed_count > 0:
        plotting.generate_pdf_report(meta, master_counts, master_counts_miz)
        print("Done. PDF generated.")
    else:
        print("No scenes processed (check thresholds/crop).")

if __name__ == '__main__':
    main()