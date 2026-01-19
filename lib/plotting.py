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

def generate_report(metadata, master_counts, mosaic_data):
    pdf_path = os.path.join(config.OUTPUT_DIR, config.PDF_FILENAME)
    print(f"Generating PDF: {pdf_path}")

    with PdfPages(pdf_path) as pdf:
        # --- PAGE 1: METADATA & MAP ---
        fig = plt.figure(figsize=(11, 12))
        
        # Text Header
        meta_txt = (
            f"VIIRS EDGE ANALYSIS REPORT\n"
            f"==========================\n"
            f"Dir:  {os.path.basename(config.DATA_DIR)}\n"
            f"Scenes: {metadata['count']}\n"
            f"Time: {metadata['start']} to {metadata['end']}\n"
            f"Exec: {metadata['duration']}\n"
            f"Lat Limit: {config.LATITUDE_THRESHOLD}\n"
        )
        plt.figtext(0.05, 0.7, meta_txt, fontsize=11, family='monospace', va='top')

        # Mosaic Map
        if HAS_CARTOPY and len(mosaic_data['lat']) > 0:
            ax = fig.add_axes([0.35, 0.05, 0.6, 0.6], projection=ccrs.NorthPolarStereo())
            ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor='darkgrey')
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            
            # Unpack mosaic
            lats = np.concatenate(mosaic_data['lat'])
            lons = np.concatenate(mosaic_data['lon'])
            clss = np.concatenate(mosaic_data['cls'])
            
            # Plot dots
            ax.scatter(lons, lats, c=clss, cmap=config.MOSAIC_CMAP, 
                       vmin=1, vmax=4, s=0.1, transform=ccrs.Geodetic())
            
            # Legend
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
                        plt.plot(bin_centers, counts/total, label=f"{key} (n={int(total)})", linewidth=2)
                
                plt.xlabel(title)
                plt.ylabel("Density")
                plt.title(f"{title} ({group_name})")
                
                # Strict Limits
                if acc_name in config.XLIMS: 
                    plt.xlim(config.XLIMS[acc_name])
                
                if has_data: plt.legend()
                else: plt.text(0.5, 0.5, "No Data", ha='center', transform=plt.gca().transAxes)
                
                plt.grid(True, linestyle="--", alpha=0.5)
                plt.tight_layout()
                pdf.savefig(); plt.close()



def save_scene_map(classification, timestamp):
    """
    Saves a PNG of the classification mask for a single scene.
    Values: 0=Skip, 1=Water, 2=Ice, 3=Cloud, 4=Land
    """
    filename = f"debug_class_{timestamp}.png"
    filepath = os.path.join(config.OUTPUT_DIR, filename)
    
    # Increase figsize to get higher resolution output
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Use 'none' interpolation to prevent aliasing/blurring of pixels
    # Use the Norm defined in config to map 0,1,2,3,4 to exact colors
    im = ax.imshow(classification, 
                   interpolation='none', 
                   cmap=config.MOSAIC_CMAP, 
                   norm=config.MOSAIC_NORM)
    
    ax.set_title(f"Scene Classification: {timestamp}\n(Black=NoData/Cut, Blue=Water, White=Ice, Gray=Cloud, Green=Land)")
    ax.axis('off')
    
    # Legend
    patches = [
        mpatches.Patch(color='black', label='No Data / Cut'),
        mpatches.Patch(color='blue',  label='Water'),
        mpatches.Patch(color='white', label='Ice'),
        mpatches.Patch(color='silver',label='Cloud'),
        mpatches.Patch(color='green', label='Land')
    ]
    ax.legend(handles=patches, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150) # Higher DPI for clearer pixels
    plt.close()