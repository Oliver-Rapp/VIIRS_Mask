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

    # --- NEW: CALCULATE COVERAGE ---
    # We subsample by 10 (::10) to speed up histogram calculation
    # We only care about VALID pixels (cls > 0)
    valid_mask = (cls > 0)
    
    # Compute 2D histogram for this scene
    # This returns a grid of counts where this scene has data
    scene_coverage, _, _ = np.histogram2d(
        data['lon'][valid_mask][::10], 
        data['lat'][valid_mask][::10], 
        bins=[config.COVERAGE_LON_BINS, config.COVERAGE_LAT_BINS]
    )
    # Convert to binary (1 = covered, 0 = not) so we count SCENES, not pixels
    scene_coverage = (scene_coverage > 0).astype(int)

    # 4. Neighbors & 5. Plotting & 6. Histograms (Keep existing logic)
    nbs = core_logic.compute_neighbors(cls, cloud_clean)
    
    if save_debug:
        plotting.generate_debug_suite(cls, nbs, ts)

    local_counts = {var: {k: np.zeros(len(config.BINS[var])-1) for k in config.KEYS} 
                    for var in config.BINS.keys()}
    
    for key in config.KEYS:
        if key not in nbs: continue
        mask = nbs[key]
        if not np.any(mask): continue
        
        for var_name in config.BINS.keys():
            if var_name not in data: continue
            vals = data[var_name][mask]
            if np.ma.is_masked(vals): vals = vals.compressed()
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                hist, _ = np.histogram(vals, bins=config.BINS[var_name])
                local_counts[var_name][key] += hist

    # Return coverage grid along with counts
    return {'counts': local_counts, 'coverage': scene_coverage}

def main():
    start_time = datetime.datetime.now()
    print("--- VIIRS PROCESSOR START ---")
    
    file_groups = data_io.get_file_groups(config.DATA_DIR)
    timestamps = sorted(list(file_groups.keys()))
    
    if config.Example_Run:
        timestamps = timestamps[:10]

    print(f"Found {len(timestamps)} scenes to process.")

    tasks = []
    for i, ts in enumerate(timestamps):
        do_debug = (i < config.NUM_DEBUG_MAPS)
        tasks.append((ts, file_groups[ts], do_debug))

    # Initialize Master Histogram Accumulator
    master_counts = {var: {k: np.zeros(len(config.BINS[var])-1) for k in config.KEYS} 
                     for var in config.BINS.keys()}
    
    # Initialize Master Coverage Grid (Zero matrix with shape of bins - 1)
    master_coverage = np.zeros((len(config.COVERAGE_LON_BINS)-1, len(config.COVERAGE_LAT_BINS)-1))

    processed_count = 0
    
    with mp.Pool(config.NUM_WORKERS) as pool:
        for result in pool.imap_unordered(process_single_scene, tasks):
            if result is None: continue
            
            processed_count += 1
            
            # Aggregate Histograms
            for var, key_dict in result['counts'].items():
                for key, hist in key_dict.items():
                    master_counts[var][key] += hist
            
            # Aggregate Coverage Map
            master_coverage += result['coverage']
            
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} scenes...")

    meta = {
        'count': processed_count,
        'duration': datetime.datetime.now() - start_time,
        'coverage_grid': master_coverage # Pass the grid to plotting
    }
    
    if processed_count > 0:
        plotting.generate_pdf_report(meta, master_counts)
        print("Done. PDF generated.")
    else:
        print("No scenes processed (check thresholds/crop).")

if __name__ == '__main__':
    main()