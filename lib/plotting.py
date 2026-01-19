import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from . import config

def generate_debug_suite(classification, neighbors, timestamp):
    """
    Generates the debug map suite.
    Updated to combine Ice and Water neighbors on a single map.
    """
    scene_dir = os.path.join(config.DEBUG_DIR, timestamp)
    if not os.path.exists(scene_dir):
        os.makedirs(scene_dir)
        
    def save_layer(active_keys, fname, title):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 1. Plot Base Map (Land, Water, Ice, Cloud)
        ax.imshow(classification, interpolation='none', 
                  cmap=config.MOSAIC_CMAP, norm=config.MOSAIC_NORM)
        
        # Create Legend for Base classes
        patches = [
            mpatches.Patch(color='black', label='NoData'),
            mpatches.Patch(color='blue', label='Water'),
            mpatches.Patch(color='white', label='Ice'),
            mpatches.Patch(color='silver', label='Cloud'),
            mpatches.Patch(color='green', label='Land')
        ]
        
        # 2. Overlay Neighbors
        # We iterate through the active_keys list to plot them on top
        for k in active_keys:
            if k in neighbors and np.any(neighbors[k]):
                color = config.NEIGHBOR_COLORS[k]
                mask = neighbors[k]
                
                # Create RGBA overlay (transparent where mask is False)
                ov = np.zeros((*mask.shape, 4))
                ov[mask] = matplotlib.colors.to_rgba(color)
                ax.imshow(ov, interpolation='none')
                
                # Add to legend with the "Pretty" name from LABEL_MAP
                label = config.LABEL_MAP.get(k, k)
                patches.append(mpatches.Patch(color=color, label=label))

        # Position legend outside to avoid blocking the map
        ax.legend(handles=patches, loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
        
        ax.set_title(f"{timestamp}\n{title}")
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(scene_dir, fname), dpi=150) # Higher DPI for clearer pixels
        plt.close()

    try:
        # --- Map 1: Base Classification Only ---
        save_layer([], "01_base_class.png", "Base Classification")
        
        # --- Map 2: Ice & Water Neighbors Combined (The Requested Map) ---
        # This list combines all 6 edge types into one plot
        combined_keys = ['iNw', 'i2Nw', 'i3Nw', 'wNi', 'w2Ni', 'w3Ni']
        save_layer(combined_keys, "02_ice_water_edges.png", "Ice & Water Neighbors")

        # --- Map 3: Cloud Neighbors & Mixed ---
        # Useful for seeing where data is being thrown out or flagged as ambiguous
        cloud_keys = ['iNc', 'i2Nc', 'wNc', 'w2Nc', 'Mixed']
        save_layer(cloud_keys, "03_cloud_mixed.png", "Cloud Neighbors & Mixed")
        
        return True
    except Exception as e:
        print(f"Plotting failed for {timestamp}: {e}")
        return False

def generate_pdf_report(meta, master_counts):
    pdf_path = os.path.join(config.OUTPUT_DIR, config.PDF_FILENAME)
    print(f"Generating Report: {pdf_path}")
    
    with PdfPages(pdf_path) as pdf:
        # ==========================================
        # PAGE 1: DETAILED METADATA & COVERAGE MAP
        # ==========================================
        fig = plt.figure(figsize=(11, 8.5)) 
        
        # --- 1. Generate Histogram Info String ---
        # Dynamically grab bin info from config so it's always accurate
        bin_info = []
        for key in ['t11', 'diff1', 'r06_tex', 't37t12_tex']:
            if key in config.BINS:
                b = config.BINS[key]
                bin_info.append(f"{key}: {len(b)-1} bins (Range: {b[0]:.1f} to {b[-1]:.1f})")
        bin_str = "\n".join(bin_info)

        # --- 2. Construct the Report Text ---
        report_txt = (
            f"VIIRS EDGE ANALYSIS REPORT\n"
            f"============================================================\n"
            f"Run ID:         {config.TIMESTAMP}\n"
            f"Source Dir:     {os.path.basename(config.DATA_DIR)}\n\n"
            
            f"DATA COVERAGE\n"
            f"------------------------------------------------------------\n"
            f"Scenes Processed: {meta['count']}\n"
            f"Data Start:       {meta['time_start']}\n"
            f"Data End:         {meta['time_end']}\n"
            f"Processing Time:  {meta['duration']}\n\n"
            
            f"GEOGRAPHIC & SOLAR CONSTRAINTS\n"
            f"------------------------------------------------------------\n"
            f"Bounding Box:     Lat {config.CROP_BOUNDS['lat_min']} to {config.CROP_BOUNDS['lat_max']}\n"
            f"                  Lon {config.CROP_BOUNDS['lon_min']} to {config.CROP_BOUNDS['lon_max']}\n"
            f"Max Solar Zenith: {config.MAX_SOLAR_ZENITH}° (Daylight limit)\n"
            f"Lat Threshold:    > {config.LATITUDE_THRESHOLD}°N (Fallback if crop disabled)\n\n"
            
            f"PROCESSING PARAMETERS\n"
            f"------------------------------------------------------------\n"
            f"Min Ice Pixels:   {config.MIN_ICE_PIXELS} px (Scene skip threshold)\n"
            f"Feature Logic:    Morphological Dilation (3x3 Struct)\n"
            f"Cloud Handling:   Clouds over land suppressed to prevent false edges\n\n"
            
            f"HISTOGRAM BINNING (Visualisation Context)\n"
            f"------------------------------------------------------------\n"
            f"{bin_str}\n"
            f"Density Plot:     Normalized (Area under curve sums to 1)"
        )
        
        # Plot Text on the left/top
        plt.figtext(0.05, 0.95, report_txt, fontfamily='monospace', fontsize=9, va='top')

        # --- 3. Plot Coverage Map (Moved down to accommodate text) ---
        proj = ccrs.NorthPolarStereo()
        # Adjusted position: [left, bottom, width, height]
        ax = fig.add_axes([0.45, 0.05, 0.5, 0.8], projection=proj)
        
        extent = [
            config.CROP_BOUNDS['lon_min'] - 5,
            config.CROP_BOUNDS['lon_max'] + 5,
            config.CROP_BOUNDS['lat_min'] - 2,
            config.CROP_BOUNDS['lat_max'] + 2
        ]
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        ax.add_feature(cfeature.LAND, facecolor='darkgrey', zorder=1)
        ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
        ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5, zorder=3)

        grid_data = meta['coverage_grid'].T
        masked_grid = np.ma.masked_where(grid_data == 0, grid_data)
        
        mesh = ax.pcolormesh(
            config.COVERAGE_LON_BINS, 
            config.COVERAGE_LAT_BINS, 
            masked_grid,
            transform=ccrs.PlateCarree(),
            cmap='plasma',
            zorder=2,
            alpha=0.8
        )
        
        # Add Crop Box
        lons = [config.CROP_BOUNDS['lon_min'], config.CROP_BOUNDS['lon_max'], 
                config.CROP_BOUNDS['lon_max'], config.CROP_BOUNDS['lon_min'], 
                config.CROP_BOUNDS['lon_min']]
        lats = [config.CROP_BOUNDS['lat_min'], config.CROP_BOUNDS['lat_min'], 
                config.CROP_BOUNDS['lat_max'], config.CROP_BOUNDS['lat_max'], 
                config.CROP_BOUNDS['lat_min']]
        
        ax.plot(lons, lats, color='black', linewidth=2, transform=ccrs.Geodetic())
        
        cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label('Scene Overlap Count')
        ax.set_title("Geographic Coverage Density")
        
        pdf.savefig()
        plt.close()

        # ==========================================
        # PAGES 2+: HISTOGRAMS (Standard Logic)
        # ==========================================
        for var_name, title in config.PLOT_ORDER:
            bins = config.BINS[var_name]
            bin_centers = (bins[:-1] + bins[1:]) / 2
            bin_width = bins[1] - bins[0]
            
            for group_name, keys in config.PLOT_GROUPS.items():
                plt.figure(figsize=(10, 6))
                has_data = False
                
                for k in keys:
                    count_hist = master_counts[var_name][k]
                    total_count = np.sum(count_hist)
                    
                    if total_count > 0:
                        has_data = True
                        density = count_hist / (total_count * bin_width)
                        
                        label = config.LABEL_MAP.get(k, k)
                        plt.plot(bin_centers, density, 
                                 label=label, 
                                 color=config.NEIGHBOR_COLORS.get(k, 'black'),
                                 linewidth=2)

                plt.title(f"{title} ({group_name})")
                plt.xlabel(title)
                plt.ylabel("Density")
                
                if var_name in config.XLIMS:
                    plt.xlim(config.XLIMS[var_name])
                
                if has_data:
                    plt.legend()
                else:
                    plt.text(0.5, 0.5, "No Data", ha='center', transform=plt.gca().transAxes)
                    
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.tight_layout()
                pdf.savefig()
                plt.close()