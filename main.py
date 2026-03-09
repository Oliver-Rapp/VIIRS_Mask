import argparse
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

    # --- SATZEN ANALYSIS ---
    local_satzen_hist = None
    local_satzen_2d   = None
    local_satzen_hist_miz = None
    local_satzen_2d_miz   = None

    if config.ENABLE_SATZEN_ANALYSIS:
        range_labels = [lbl for _, _, lbl in config.SATZEN_RANGES]

        local_satzen_hist = {
            lbl: {dv: {cls: np.zeros(len(config.BINS[dv]) - 1)
                       for cls in config.SATZEN_ANALYSIS_CLASSES}
                  for dv in config.SATZEN_DIFF_VARS}
            for lbl in range_labels
        }
        local_satzen_2d = {
            dv: {cls: np.zeros((len(config.BINS['satz']) - 1, len(config.BINS[dv]) - 1))
                 for cls in config.SATZEN_ANALYSIS_CLASSES}
            for dv in config.SATZEN_DIFF_VARS
        }

        if config.ENABLE_MIZ_HISTOGRAMS and miz_mask is not None:
            local_satzen_hist_miz = {
                lbl: {dv: {cls: np.zeros(len(config.BINS[dv]) - 1)
                           for cls in config.SATZEN_ANALYSIS_CLASSES}
                      for dv in config.SATZEN_DIFF_VARS}
                for lbl in range_labels
            }
            local_satzen_2d_miz = {
                dv: {cls: np.zeros((len(config.BINS['satz']) - 1, len(config.BINS[dv]) - 1))
                     for cls in config.SATZEN_ANALYSIS_CLASSES}
                for dv in config.SATZEN_DIFF_VARS
            }

        satz_arr = data['satz']
        if np.ma.is_masked(satz_arr):
            satz_arr = satz_arr.filled(np.nan)

        for cls_key in config.SATZEN_ANALYSIS_CLASSES:
            if cls_key not in nbs: continue
            cls_mask_arr = nbs[cls_key]
            if not np.any(cls_mask_arr): continue

            sz = satz_arr[cls_mask_arr]

            for dv in config.SATZEN_DIFF_VARS:
                if dv not in data: continue

                dv_arr = data[dv]
                if np.ma.is_masked(dv_arr):
                    dv_arr = dv_arr.filled(np.nan)
                diff = dv_arr[cls_mask_arr]

                valid = ~np.isnan(sz) & ~np.isnan(diff)
                sz_c   = sz[valid]
                diff_c = diff[valid]
                if len(sz_c) == 0: continue

                # 2D histogram
                h2d, _, _ = np.histogram2d(sz_c, diff_c,
                                           bins=[config.BINS['satz'], config.BINS[dv]])
                local_satzen_2d[dv][cls_key] += h2d

                # 1D histograms per satzen range
                ranges = config.SATZEN_RANGES
                for i, (lo, hi, lbl) in enumerate(ranges):
                    last = (i == len(ranges) - 1)
                    rng = (sz_c >= lo) & (sz_c <= hi if last else sz_c < hi)
                    if np.any(rng):
                        h1d, _ = np.histogram(diff_c[rng], bins=config.BINS[dv])
                        local_satzen_hist[lbl][dv][cls_key] += h1d

                # MIZ subset
                if local_satzen_hist_miz is not None and miz_mask is not None:
                    miz_cls = cls_mask_arr & miz_mask
                    if np.any(miz_cls):
                        sz_m   = satz_arr[miz_cls]
                        diff_m = dv_arr[miz_cls]
                        valid_m = ~np.isnan(sz_m) & ~np.isnan(diff_m)
                        sz_mc   = sz_m[valid_m]
                        diff_mc = diff_m[valid_m]
                        if len(sz_mc) > 0:
                            h2d_m, _, _ = np.histogram2d(sz_mc, diff_mc,
                                                         bins=[config.BINS['satz'], config.BINS[dv]])
                            local_satzen_2d_miz[dv][cls_key] += h2d_m
                            for i, (lo, hi, lbl) in enumerate(ranges):
                                last = (i == len(ranges) - 1)
                                rng_m = (sz_mc >= lo) & (sz_mc <= hi if last else sz_mc < hi)
                                if np.any(rng_m):
                                    h1d_m, _ = np.histogram(diff_mc[rng_m], bins=config.BINS[dv])
                                    local_satzen_hist_miz[lbl][dv][cls_key] += h1d_m

    return {
        'ts': ts,
        'counts': local_counts,
        'counts_miz': local_counts_miz,
        'coverage': scene_coverage,
        'satzen_hist': local_satzen_hist,
        'satzen_2d':   local_satzen_2d,
        'satzen_hist_miz': local_satzen_hist_miz,
        'satzen_2d_miz':   local_satzen_2d_miz,
    }

def main():
    parser = argparse.ArgumentParser(description="VIIRS MIZ Edge Analyser")
    parser.add_argument('--data-dir',   default=None,
                        help=f"Directory of input NetCDF files (default: DATA_DIR in config.py)")
    parser.add_argument('--output-dir', default=None,
                        help=f"Directory for PDF report and debug maps (default: OUTPUT_DIR in config.py)")
    parser.add_argument('--workers',    default=None, type=int,
                        help="Number of parallel worker processes (default: all cores)")
    args = parser.parse_args()

    if args.data_dir:
        config.DATA_DIR = args.data_dir
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    if args.workers is not None:
        config.NUM_WORKERS = args.workers

    start_time = datetime.datetime.now()
    print("--- VIIRS PROCESSOR START ---")
    print(f"  Data dir:   {config.DATA_DIR}")
    print(f"  Output dir: {config.OUTPUT_DIR}")

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

    master_satzen_hist = None
    master_satzen_2d   = None
    master_satzen_hist_miz = None
    master_satzen_2d_miz   = None
    if config.ENABLE_SATZEN_ANALYSIS:
        range_labels = [lbl for _, _, lbl in config.SATZEN_RANGES]
        master_satzen_hist = {
            lbl: {dv: {cls: np.zeros(len(config.BINS[dv]) - 1)
                       for cls in config.SATZEN_ANALYSIS_CLASSES}
                  for dv in config.SATZEN_DIFF_VARS}
            for lbl in range_labels
        }
        master_satzen_2d = {
            dv: {cls: np.zeros((len(config.BINS['satz']) - 1, len(config.BINS[dv]) - 1))
                 for cls in config.SATZEN_ANALYSIS_CLASSES}
            for dv in config.SATZEN_DIFF_VARS
        }
        if config.ENABLE_MIZ_HISTOGRAMS:
            master_satzen_hist_miz = {
                lbl: {dv: {cls: np.zeros(len(config.BINS[dv]) - 1)
                           for cls in config.SATZEN_ANALYSIS_CLASSES}
                      for dv in config.SATZEN_DIFF_VARS}
                for lbl in range_labels
            }
            master_satzen_2d_miz = {
                dv: {cls: np.zeros((len(config.BINS['satz']) - 1, len(config.BINS[dv]) - 1))
                     for cls in config.SATZEN_ANALYSIS_CLASSES}
                for dv in config.SATZEN_DIFF_VARS
            }

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

            if config.ENABLE_SATZEN_ANALYSIS and result['satzen_hist']:
                for lbl, dv_dict in result['satzen_hist'].items():
                    for dv, cls_dict in dv_dict.items():
                        for cls, hist in cls_dict.items():
                            master_satzen_hist[lbl][dv][cls] += hist
                for dv, cls_dict in result['satzen_2d'].items():
                    for cls, h2d in cls_dict.items():
                        master_satzen_2d[dv][cls] += h2d
                if result['satzen_hist_miz']:
                    for lbl, dv_dict in result['satzen_hist_miz'].items():
                        for dv, cls_dict in dv_dict.items():
                            for cls, hist in cls_dict.items():
                                master_satzen_hist_miz[lbl][dv][cls] += hist
                    for dv, cls_dict in result['satzen_2d_miz'].items():
                        for cls, h2d in cls_dict.items():
                            master_satzen_2d_miz[dv][cls] += h2d

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
        plotting.generate_pdf_report(
            meta, master_counts, master_counts_miz,
            master_satzen_hist, master_satzen_2d,
            master_satzen_hist_miz, master_satzen_2d_miz
        )
        print("Done. PDF generated.")
    else:
        print("No scenes processed (check thresholds/crop).")

if __name__ == '__main__':
    main()