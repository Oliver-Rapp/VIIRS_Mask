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
    scene_dir = config.DEBUG_DIR
    scene_sub_dir = os.path.join(scene_dir, timestamp)
    if not os.path.exists(scene_sub_dir):
        os.makedirs(scene_sub_dir)
        
    def save_layer(active_keys, fname, title):
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 1. Plot Base Map
        ax.imshow(classification, interpolation='none', 
                  cmap=config.MOSAIC_CMAP, norm=config.MOSAIC_NORM)
        
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
                
                ov = np.zeros((*mask.shape, 4))
                ov[mask] = matplotlib.colors.to_rgba(color)
                ax.imshow(ov, interpolation='none')
                
                label = config.LABEL_MAP.get(k, k)
                patches.append(mpatches.Patch(color=color, label=label))

        ax.legend(handles=patches, loc='upper left', bbox_to_anchor=(1, 1), fontsize='x-small')
        
        ax.set_title(f"{timestamp}\n{title}")
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(scene_sub_dir, fname), dpi=150)
        plt.close()

    try:
        save_layer([], "01_base_class.png", "Base Classification")
        
        combined_keys = [
            'IN', 'WN', 'CN', 
            'I3nW', 'I3nC', 'W3nI', 'W3nC', 'C3nI', 'C3nW', 
            'I2nW', 'I2nC', 'W2nI', 'W2nC', 'C2nI', 'C2nW', 
            'InW', 'InC', 'WnI', 'WnC', 'CnI', 'CnW',       
            'Mixed'                                         
        ]
        save_layer(combined_keys, "02_combined_analysis.png", "Full Classification (Interior, Neighbors, Mixed)")
        
        return True
    except Exception as e:
        print(f"Plotting failed for {timestamp}: {e}")
        return False

def _plot_histogram_pages(pdf, counts_dict, title_prefix=""):
    """
    Helper function to plot a set of histograms into the PDF.
    """
    for var_name, title in config.PLOT_ORDER:
        bins = config.BINS[var_name]
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_width = bins[1] - bins[0]
        
        for group_name, keys in config.PLOT_GROUPS.items():
            plt.figure(figsize=(10, 6))
            has_data = False
            
            for k in keys:
                if k not in counts_dict[var_name]: continue
                
                count_hist = counts_dict[var_name][k]
                total_count = np.sum(count_hist)
                
                if total_count > 0:
                    has_data = True
                    density = count_hist / (total_count * bin_width)
                    
                    label = config.LABEL_MAP.get(k, k)
                    plt.plot(bin_centers, density, 
                             label=label, 
                             color=config.NEIGHBOR_COLORS.get(k, 'black'),
                             linewidth=2)

            full_title = f"{title_prefix}{title} ({group_name})"
            plt.title(full_title)
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

def generate_pdf_report(meta, master_counts, master_counts_miz=None):
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

        miz_str = "Disabled"
        if config.ENABLE_MIZ_HISTOGRAMS:
            miz_str = f"Active ({config.MIZ_T11_RANGE[0]}K - {config.MIZ_T11_RANGE[1]}K)"

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
            f"MIZ SUBSET CONFIGURATION\n"
            f"------------------------------------------------------------\n"
            f"Filtering:        {miz_str}\n\n"
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
            cmap='plasma', zorder=2, alpha=0.8
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

        # PART 1: Standard Plots
        _plot_histogram_pages(pdf, master_counts, title_prefix="")

        # PART 2: MIZ Subset Plots
        if config.ENABLE_MIZ_HISTOGRAMS and master_counts_miz:
            # Add a separator page
            plt.figure(figsize=(11, 8.5))
            plt.text(0.5, 0.5, 
                     f"MIZ SUBSET ANALYSIS\n\nTemperature Filter: {config.MIZ_T11_RANGE[0]}K < T11 < {config.MIZ_T11_RANGE[1]}K", 
                     ha='center', va='center', fontsize=20)
            plt.axis('off')
            pdf.savefig()
            plt.close()
            
            _plot_histogram_pages(pdf, master_counts_miz, title_prefix="[MIZ Subset] ")