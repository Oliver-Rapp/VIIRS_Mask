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
    Generates the 6-map debug suite exactly as requested.
    """
    scene_dir = os.path.join(config.DEBUG_DIR, timestamp)
    if not os.path.exists(scene_dir):
        os.makedirs(scene_dir)
        
    def save_layer(active_keys, fname, title):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Base map
        ax.imshow(classification, interpolation='none', 
                  cmap=config.MOSAIC_CMAP, norm=config.MOSAIC_NORM)
        
        patches = [
            mpatches.Patch(color='black', label='NoData'),
            mpatches.Patch(color='blue', label='Water'),
            mpatches.Patch(color='white', label='Ice'),
            mpatches.Patch(color='silver', label='Cloud'),
            mpatches.Patch(color='green', label='Land')
        ]
        
        # Overlay neighbors
        for k in active_keys:
            if k in neighbors and np.any(neighbors[k]):
                color = config.NEIGHBOR_COLORS[k]
                mask = neighbors[k]
                
                # Make RGBA overlay
                ov = np.zeros((*mask.shape, 4))
                ov[mask] = matplotlib.colors.to_rgba(color)
                ax.imshow(ov, interpolation='none')
                
                # Use mapped label for legend
                label = config.LABEL_MAP.get(k, k)
                patches.append(mpatches.Patch(color=color, label=label))

        ax.legend(handles=patches, loc='lower right', fontsize='small')
        ax.set_title(f"{timestamp}\n{title}")
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(scene_dir, fname), dpi=100)
        plt.close()

    try:
        # 1. Base
        save_layer([], "01_base.png", "Classification Map")
        
        # 2. Ice Neighbors (Red/Orange/Gold)
        save_layer(['iNw', 'i2Nw', 'i3Nw'], 
                   "02_ice_neighbors.png", "Ice Neighbors")
        
        # 3. Water Neighbors (Cyan/Navy/Purple)
        save_layer(['wNi', 'w2Ni', 'w3Ni'], 
                   "03_water_neighbors.png", "Water Neighbors")

        # 4. Cloud Neighbors (Grays) with Mixed
        save_layer(['iNc', 'i2Nc', 'wNc', 'w2Nc', 'Mixed'], 
                   "04_cloud_mixed.png", "Cloud Neighbors with Mixed")
        
        # 5. Mixed Only
        save_layer(['Mixed'], "05_mixed_only.png", "Only Mixed Neighbors")
        
        # 6. Combined Ice/Water (The "Rainbow" Edge)
        save_layer(['iNw', 'i2Nw', 'i3Nw', 'wNi', 'w2Ni', 'w3Ni'], 
                   "06_ice_water_combined.png", "Ice & Water Neighbors")
        return True
    except Exception as e:
        print(f"Plotting failed for {timestamp}: {e}")
        return False

def generate_pdf_report(meta, master_counts):
    pdf_path = os.path.join(config.OUTPUT_DIR, config.PDF_FILENAME)
    print(f"Generating Report: {pdf_path}")
    
    with PdfPages(pdf_path) as pdf:
        # ==========================================
        # PAGE 1: METADATA & COVERAGE MAP
        # ==========================================
        fig = plt.figure(figsize=(11, 8)) # Wider landscape figure
        
        # --- Top: Text Metadata ---
        txt = (
            f"VIIRS ANALYSIS REPORT\n"
            f"=====================\n"
            f"Processed Scenes: {meta['count']}\n"
            f"Total Duration:   {meta['duration']}\n"
            f"Lat Threshold:    {config.LATITUDE_THRESHOLD}\n"
            f"Min Ice Pixels:   {config.MIN_ICE_PIXELS}"
        )
        plt.figtext(0.05, 0.95, txt, fontfamily='monospace', fontsize=10, va='top')

        # --- Bottom: Coverage Map ---
        # Define Projection (North Polar Stereo)
        proj = ccrs.NorthPolarStereo()
        ax = fig.add_axes([0.1, 0.05, 0.8, 0.75], projection=proj)
        
        # 1. Setup Map Extents (Zoom to crop bounds + margin)
        extent = [
            config.CROP_BOUNDS['lon_min'] - 5,
            config.CROP_BOUNDS['lon_max'] + 5,
            config.CROP_BOUNDS['lat_min'] - 2,
            config.CROP_BOUNDS['lat_max'] + 2
        ]
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        # 2. Add Geographic Features
        ax.add_feature(cfeature.LAND, facecolor='darkgrey', zorder=1)
        ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
        ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5, zorder=3)

        # 3. Plot Data Density Heatmap
        # Using pcolormesh with the pre-calculated bins
        # Mask zeros to make them transparent
        grid_data = meta['coverage_grid'].T  # Transpose needed for pcolormesh usually
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
        
        # Colorbar
        cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
        cbar.set_label('Number of Overlapping Scenes')

        # 4. Draw the Bounding Box
        # Create a rectangle polygon for the config bounds
        lons = [config.CROP_BOUNDS['lon_min'], config.CROP_BOUNDS['lon_max'], 
                config.CROP_BOUNDS['lon_max'], config.CROP_BOUNDS['lon_min'], 
                config.CROP_BOUNDS['lon_min']]
        lats = [config.CROP_BOUNDS['lat_min'], config.CROP_BOUNDS['lat_min'], 
                config.CROP_BOUNDS['lat_max'], config.CROP_BOUNDS['lat_max'], 
                config.CROP_BOUNDS['lat_min']]
        
        ax.plot(lons, lats, color='black', linewidth=2, transform=ccrs.Geodetic(), label='Crop Box')
        ax.legend(loc='upper right')
        
        ax.set_title("Data Coverage Density")
        
        pdf.savefig()
        plt.close()

        # ==========================================
        # PAGES 2+: HISTOGRAMS (Existing Logic)
        # ==========================================
        for var_name, title in config.PLOT_ORDER:
            # ... [Same histogram plotting code as before] ...
            # (Copy the histogram loop from previous response here)
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