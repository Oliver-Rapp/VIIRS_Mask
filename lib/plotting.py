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
    """
    # Use the timestamped run directory
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
        for k in active_keys:
            if k in neighbors and np.any(neighbors[k]):
                color = config.NEIGHBOR_COLORS[k]
                mask = neighbors[k]
                
                # Create RGBA overlay
                ov = np.zeros((*mask.shape, 4))
                ov[mask] = matplotlib.colors.to_rgba(color)
                ax.imshow(ov, interpolation='none')
                
                label = config.LABEL_MAP.get(k, k)
                patches.append(mpatches.Patch(color=color, label=label))

        ax.legend(handles=patches, loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
        
        ax.set_title(f"{timestamp}\n{title}")
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(scene_dir, fname), dpi=150)
        plt.close()

    try:
        # Map 1: Base Classification
        save_layer([], "01_base_class.png", "Base Classification")
        
        # Map 2: Edges (Ice/Water)
        edge_keys = ['iNw', 'i2Nw', 'i3Nw', 'wNi', 'w2Ni', 'w3Ni']
        save_layer(edge_keys, "02_ice_water_edges.png", "Ice & Water Neighbors")

        # Map 3: Cloud & Mixed
        cloud_keys = ['iNc', 'i2Nc', 'wNc', 'w2Nc', 'Mixed']
        save_layer(cloud_keys, "03_cloud_mixed.png", "Cloud Neighbors & Mixed")

        # Map 4: Interior/Pure Classes
        interior_keys = ['IN', 'WN', 'CN']
        save_layer(interior_keys, "04_interior_classes.png", "Interior / No Neighbors")
        
        return True
    except Exception as e:
        print(f"Plotting failed for {timestamp}: {e}")
        return False

def generate_pdf_report(meta, master_counts):
    pdf_path = os.path.join(config.OUTPUT_DIR, config.PDF_FILENAME)
    print(f"Generating Report: {pdf_path}")
    
    with PdfPages(pdf_path) as pdf:
        # PAGE 1: METADATA & COVERAGE
        fig = plt.figure(figsize=(11, 8.5)) 
        
        bin_info = []
        for key in ['t11', 'diff1', 'r06_tex', 't37t12_tex']:
            if key in config.BINS:
                b = config.BINS[key]
                bin_info.append(f"{key}: {len(b)-1} bins (Range: {b[0]:.1f} to {b[-1]:.1f})")
        bin_str = "\n".join(bin_info)

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
            f"Max Solar Zenith: {config.MAX_SOLAR_ZENITH}°\n\n"
            f"HISTOGRAM BINNING\n"
            f"------------------------------------------------------------\n"
            f"{bin_str}\n"
        )
        
        plt.figtext(0.05, 0.95, report_txt, fontfamily='monospace', fontsize=9, va='top')

        proj = ccrs.NorthPolarStereo()
        ax = fig.add_axes([0.45, 0.05, 0.5, 0.8], projection=proj)
        
        margin_lon = 5; margin_lat = 2
        extent = [
            config.CROP_BOUNDS['lon_min'] - margin_lon,
            config.CROP_BOUNDS['lon_max'] + margin_lon,
            config.CROP_BOUNDS['lat_min'] - margin_lat,
            config.CROP_BOUNDS['lat_max'] + margin_lat
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
        
        bounds = config.CROP_BOUNDS
        n_steps = 100
        
        l1_x = np.linspace(bounds['lon_min'], bounds['lon_max'], n_steps)
        l1_y = np.full(n_steps, bounds['lat_min'])
        l2_x = np.full(n_steps, bounds['lon_max'])
        l2_y = np.linspace(bounds['lat_min'], bounds['lat_max'], n_steps)
        l3_x = np.linspace(bounds['lon_max'], bounds['lon_min'], n_steps)
        l3_y = np.full(n_steps, bounds['lat_max'])
        l4_x = np.full(n_steps, bounds['lon_min'])
        l4_y = np.linspace(bounds['lat_max'], bounds['lat_min'], n_steps)
        
        box_lons = np.concatenate([l1_x, l2_x, l3_x, l4_x])
        box_lats = np.concatenate([l1_y, l2_y, l3_y, l4_y])
        
        ax.plot(box_lons, box_lats, color='black', linewidth=2, 
                transform=ccrs.PlateCarree(), label='Analysis Region', zorder=10)
        
        cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label('Scene Overlap Count')
        ax.set_title("Geographic Coverage Density")
        
        pdf.savefig()
        plt.close()

        # PAGES 2+: HISTOGRAMS
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