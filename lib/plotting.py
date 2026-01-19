import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from . import config

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

def generate_debug_suite(classification, neighbors, timestamp):
    """
    Generates a full suite of debug maps for a single scene.
    Creates a subfolder for the scene and saves 6 different visualizations.
    """
    # Create a subfolder for this specific scene
    scene_dir = os.path.join(config.DEBUG_DIR, timestamp)
    if not os.path.exists(scene_dir):
        os.makedirs(scene_dir)

    # Helper function to create one specific map overlay
    def plot_layer(active_keys, filename, title):
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 1. Base Classification (Black/Blue/White/Gray/Green)
        ax.imshow(classification, interpolation='none', 
                  cmap=config.MOSAIC_CMAP, norm=config.MOSAIC_NORM)
        
        legend_patches = [
            mpatches.Patch(color='black', label='No Data'),
            mpatches.Patch(color='blue',  label='Water'),
            mpatches.Patch(color='white', label='Ice'),
            mpatches.Patch(color='silver',label='Cloud'),
            mpatches.Patch(color='green', label='Land')
        ]

        # 2. Overlay Neighbors
        # We iterate through a specific draw order to ensure consistency
        # Order: Cloud NB -> Water NB -> Ice NB -> Mixed (Top)
        draw_order = ['iNc','wNc','i2Nc','w2Nc', 
                      'w3Ni','w2Ni','wNi', 
                      'i3Nw','i2Nw','iNw', 
                      'Mixed']
        
        for key in draw_order:
            if key in active_keys and key in neighbors:
                mask = neighbors[key]
                if np.any(mask):
                    color = config.NEIGHBOR_COLORS[key]
                    
                    # Create a transparent overlay where mask is False
                    masked_overlay = np.ma.masked_where(~mask, np.ones_like(classification))
                    
                    # Create a solid color overlay
                    cmap_overlay = matplotlib.colors.ListedColormap([color])
                    
                    # Plot it
                    ax.imshow(masked_overlay, interpolation='none', cmap=cmap_overlay, alpha=1.0)
                    legend_patches.append(mpatches.Patch(color=color, label=key))

        ax.set_title(f"Scene: {timestamp}\n{title}")
        ax.axis('off')
        
        # Place legend outside to not obscure data
        ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        
        plt.tight_layout()
        plt.savefig(os.path.join(scene_dir, filename), dpi=100)
        plt.close()

    # --- GENERATE THE 6 MAPS ---
    
    # 1. Base Map
    plot_layer([], "01_classification.png", "Base Classification")

    # 2. Ice/Water Interaction
    plot_layer(['iNw', 'i2Nw', 'i3Nw', 'wNi', 'w2Ni', 'w3Ni'], 
               "02_ice_water_neighbors.png", "Ice/Water Neighbors")

    # 3. Cloud Interaction
    plot_layer(['iNc', 'i2Nc', 'wNc', 'w2Nc', 'Mixed'], 
               "03_cloud_mixed_neighbors.png", "Cloud Neighbors & Mixed")

    # 4. Ice + Mixed
    plot_layer(['iNw', 'i2Nw', 'i3Nw', 'Mixed'], 
               "04_ice_mixed.png", "Ice Neighbors & Mixed")

    # 5. Water + Mixed
    plot_layer(['wNi', 'w2Ni', 'w3Ni', 'Mixed'], 
               "05_water_mixed.png", "Water Neighbors & Mixed")

    # 6. Only Mixed
    plot_layer(['Mixed'], 
               "06_only_mixed.png", "Only Mixed Neighbors")


def generate_report(metadata, master_counts, mosaic_data):
    """
    Generates the final PDF with Metadata, Mosaic Map, and Histograms.
    """
    pdf_path = os.path.join(config.OUTPUT_DIR, config.PDF_FILENAME)
    print(f"Generating PDF: {pdf_path}")

    with PdfPages(pdf_path) as pdf:
        # --- PAGE 1: METADATA & MOSAIC MAP ---
        fig = plt.figure(figsize=(11, 12))
        
        # Formatted Metadata Header
        meta_txt = (
            f"VIIRS EDGE ANALYSIS REPORT\n"
            f"=====================\n"
            f"Run ID:         {config.TIMESTAMP}\n"
            f"Source Dir:     {os.path.basename(config.DATA_DIR)}\n"
            f"Execution Time: {metadata['duration']}\n\n"
            f"DATA COVERAGE\n"
            f"-------------\n"
            f"Scenes Found:   {metadata['count']}\n"
            f"Data Start:     {metadata['start']}\n"
            f"Data End:       {metadata['end']}\n\n"
            f"FILTERS & THRESHOLDS\n"
            f"--------------------\n"
            f"Latitude Limit: > {config.LATITUDE_THRESHOLD}°N\n"
            f"Sun Zenith Max: {config.MAX_SOLAR_ZENITH if config.MAX_SOLAR_ZENITH else 'None'}°\n"
            f"Min Ice Pixels: > {config.MIN_ICE_PIXELS}\n"
            f"Crop Enabled:   {config.USE_CROP}\n"
        )
        if config.USE_CROP:
            meta_txt += f"Crop Bounds:    {config.CROP_BOUNDS}\n"
        
        plt.figtext(0.05, 0.60, meta_txt, fontsize=10, family='monospace', va='top')

        # Mosaic Map (Using Cartopy if available)
        if HAS_CARTOPY and len(mosaic_data['lat']) > 0:
            ax = fig.add_axes([0.1, 0.05, 0.8, 0.55], projection=ccrs.NorthPolarStereo())
            ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor='darkgrey')
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            
            # Concatenate list of arrays into single arrays for plotting
            lats = np.concatenate(mosaic_data['lat'])
            lons = np.concatenate(mosaic_data['lon'])
            clss = np.concatenate(mosaic_data['cls'])
            
            # Plot the dots
            ax.scatter(lons, lats, c=clss, cmap=config.MOSAIC_CMAP, 
                       vmin=0, vmax=4, s=0.1, transform=ccrs.Geodetic())
            
            # Map Legend
            patches = [
                mpatches.Patch(color='blue', label='Water'),
                mpatches.Patch(color='white', label='Ice'),
                mpatches.Patch(color='silver', label='Cloud'),
                mpatches.Patch(color='green', label='Land')
            ]
            ax.legend(handles=patches, loc='lower left', title="Class")
            ax.set_title(f"Mosaic Coverage (N={metadata['count']})")
        
        pdf.savefig(fig); plt.close()

        # --- PAGES 2+: HISTOGRAMS ---
        for acc_name, title in config.PLOT_ORDER:
            bins = config.BINS[acc_name]
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            for group_name, keys in config.PLOT_GROUPS.items():
                plt.figure(figsize=(10, 6))
                has_data = False
                
                for key in keys:
                    counts = master_counts[acc_name][key]
                    total = np.sum(counts)
                    if total > 0:
                        has_data = True
                        # Plot density (counts / total)
                        plt.plot(bin_centers, counts/total, label=f"{key} (n={int(total)})", linewidth=2)
                
                plt.xlabel(title)
                plt.ylabel("Density")
                plt.title(f"{title} ({group_name})")
                
                # Apply Strict Limits if defined in config
                if acc_name in config.XLIMS: 
                    plt.xlim(config.XLIMS[acc_name])
                
                if has_data: plt.legend()
                else: plt.text(0.5, 0.5, "No Data", ha='center', transform=plt.gca().transAxes)
                
                plt.grid(True, linestyle="--", alpha=0.5)
                plt.tight_layout()
                pdf.savefig(); plt.close()