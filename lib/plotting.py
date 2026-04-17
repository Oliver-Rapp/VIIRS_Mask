import os
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from . import config

def _format_date_range(ts_start, ts_end):
    """
    Parses timestamps like '20250415T105803Z' and returns a human-readable
    date range string, e.g. '2025-04-01 – 2025-04-14'.
    Falls back gracefully if parsing fails.
    """
    def _parse_date(ts):
        try:
            return datetime.datetime.strptime(str(ts)[:8], '%Y%m%d').strftime('%Y-%m-%d')
        except (ValueError, TypeError):
            return str(ts)

    if ts_start == "N/A" or ts_end == "N/A":
        return "N/A"
    return f"{_parse_date(ts_start)} – {_parse_date(ts_end)}"

def generate_debug_suite(classification, neighbors, timestamp, miz_mask=None):
    """
    Generates the debug map suite.
    """
    scene_dir = config.DEBUG_DIR
    scene_sub_dir = os.path.join(scene_dir, timestamp)
    if not os.path.exists(scene_sub_dir):
        os.makedirs(scene_sub_dir)
        
    def save_layer(active_keys, fname, title, extra_mask=None, extra_color=None, extra_label=None):
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 1. Plot Base Map
        alpha = 0.6 if extra_mask is not None else 1.0
        
        ax.imshow(classification, interpolation='none', 
                  cmap=config.MOSAIC_CMAP, norm=config.MOSAIC_NORM, alpha=alpha)
        
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

        # 3. Overlay Extra Mask (e.g. MIZ Subset)
        if extra_mask is not None and np.any(extra_mask):
            ov = np.zeros((*extra_mask.shape, 4))
            ov[extra_mask] = matplotlib.colors.to_rgba(extra_color)
            ax.imshow(ov, interpolation='none')
            patches.append(mpatches.Patch(color=extra_color, label=extra_label))

        ax.legend(handles=patches, loc='upper left', bbox_to_anchor=(1, 1), fontsize='x-small')
        
        ax.set_title(f"{timestamp}\n{title}")
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(scene_sub_dir, fname), dpi=150)
        plt.close()

    try:
        # Map 1: Base Classification
        save_layer([], "01_base_class.png", "Base Classification")
        
        # Map 2: The Combined Map (Interior + Neighbors + Mixed)
        combined_keys = [
            'IN', 'WN', 'CN', 
            'I3nW', 'I3nC', 'W3nI', 'W3nC', 'C3nI', 'C3nW', 
            'I2nW', 'I2nC', 'W2nI', 'W2nC', 'C2nI', 'C2nW', 
            'InW', 'InC', 'WnI', 'WnC', 'CnI', 'CnW',       
            'Mixed'                                         
        ]
        save_layer(combined_keys, "02_combined_analysis.png", "Full Classification")

        # Map 5: MIZ Diagnostic (New)
        if miz_mask is not None:
            t_range_str = f"{config.MIZ_T11_RANGE[0]}K-{config.MIZ_T11_RANGE[1]}K"
            save_layer([], "05_miz_diagnostic.png", 
                       f"MIZ Subset Diagnostic ({t_range_str})",
                       extra_mask=miz_mask,
                       extra_color='gold',
                       extra_label='Pixels in Range')
        
        return True
    except Exception as e:
        print(f"Plotting failed for {timestamp}: {e}")
        return False

def _format_count(n):
    """Helper to format large numbers (e.g. 1.2M, 45.5K)"""
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    else:
        return str(int(n))

def _plot_histogram_pages(pdf, counts_dict, title_prefix="", png_dir=None, date_tag=""):
    for var_name, title in config.PLOT_ORDER:
        bins = config.BINS[var_name]
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_width = bins[1] - bins[0]

        for group_name, keys in config.PLOT_GROUPS.items():
            fig = plt.figure(figsize=(10, 6))
            has_data = False

            for k in keys:
                if k not in counts_dict[var_name]: continue

                count_hist = counts_dict[var_name][k]
                total_count = np.sum(count_hist)

                if total_count > 0:
                    has_data = True
                    # Normalized density
                    density = count_hist / (total_count * bin_width)

                    label_text = config.LABEL_MAP.get(k, k)
                    count_str = _format_count(total_count)
                    full_label = f"{label_text} ({count_str})"

                    plt.plot(bin_centers, density,
                             label=full_label,
                             color=config.NEIGHBOR_COLORS.get(k, 'black'),
                             linewidth=2)

            full_title = f"{title_prefix}{title} ({group_name})"
            plt.title(full_title)
            plt.xlabel(title)
            plt.ylabel("Density")

            if var_name in config.XLIMS:
                plt.xlim(config.XLIMS[var_name])

            if has_data:
                # Place legend, now containing N counts
                plt.legend()
            else:
                plt.text(0.5, 0.5, "No Data", ha='center', transform=plt.gca().transAxes)

            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            pdf.savefig()
            if png_dir:
                slug = group_name.lower().replace(' ', '_')
                fname = f"{date_tag}_{var_name}_{slug}.png" if date_tag else f"{var_name}_{slug}.png"
                fig.savefig(os.path.join(png_dir, fname), dpi=300, bbox_inches='tight')
            plt.close()

def _plot_satzen_histogram_pages(pdf, satzen_hist, title_prefix="", png_dir=None, date_tag=""):
    """
    For each diff variable: one figure with one subplot per class (IN, WN).
    Each subplot shows 3 density lines — one per satzen range.
    """
    var_titles = {v: t for v, t in config.PLOT_ORDER}

    for dv in config.SATZEN_DIFF_VARS:
        if dv not in var_titles or dv not in config.BINS:
            continue

        bins = config.BINS[dv]
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_width    = bins[1] - bins[0]

        n_cls = len(config.SATZEN_ANALYSIS_CLASSES)
        fig, axes = plt.subplots(1, n_cls, figsize=(6 * n_cls, 5), sharey=False)
        if n_cls == 1:
            axes = [axes]

        for ax, cls_key in zip(axes, config.SATZEN_ANALYSIS_CLASSES):
            ax_has_data = False
            for (lo, hi, lbl), color in zip(config.SATZEN_RANGES, config.SATZEN_RANGE_COLORS):
                count_hist = satzen_hist[lbl][dv][cls_key]
                total = np.sum(count_hist)
                if total > 0:
                    density = count_hist / (total * bin_width)
                    ax.plot(bin_centers, density,
                            label=f"{lbl} ({_format_count(total)})",
                            color=color, linewidth=2)
                    ax_has_data = True

            ax.set_title(cls_key)
            ax.set_xlabel(var_titles[dv])
            ax.set_ylabel("Density")
            if dv in config.XLIMS:
                ax.set_xlim(config.XLIMS[dv])
            if ax_has_data:
                ax.legend(fontsize='small')
            else:
                ax.text(0.5, 0.5, "No Data", ha='center', transform=ax.transAxes)
            ax.grid(True, linestyle='--', alpha=0.5)

        fig.suptitle(f"{title_prefix}Satzen Analysis: {var_titles[dv]}", fontsize=12, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig)
        if png_dir:
            fname = f"{date_tag}_{dv}.png" if date_tag else f"{dv}.png"
            fig.savefig(os.path.join(png_dir, fname), dpi=300, bbox_inches='tight')
        plt.close(fig)


def _plot_satzen_scatter_pages(pdf, satzen_2d, title_prefix="", png_dir=None, date_tag=""):
    """
    For each diff variable: one figure with one subplot per class (IN, WN).
    Each subplot is a 2D density heatmap: X=satellite zenith angle, Y=diff.
    Vertical dashed lines mark the satzen range boundaries.
    """
    from matplotlib.colors import LogNorm

    var_titles = {v: t for v, t in config.PLOT_ORDER}
    satz_bins = config.BINS['satz']

    for dv in config.SATZEN_DIFF_VARS:
        if dv not in var_titles or dv not in config.BINS:
            continue

        diff_bins = config.BINS[dv]

        n_cls = len(config.SATZEN_ANALYSIS_CLASSES)
        fig, axes = plt.subplots(1, n_cls, figsize=(7 * n_cls, 5))
        if n_cls == 1:
            axes = [axes]

        for ax, cls_key in zip(axes, config.SATZEN_ANALYSIS_CLASSES):
            h2d = satzen_2d[dv][cls_key]  # shape: (n_satz, n_diff)

            if np.sum(h2d) == 0:
                ax.text(0.5, 0.5, "No Data", ha='center', transform=ax.transAxes)
                ax.set_title(cls_key)
                continue

            masked = np.ma.masked_where(h2d.T == 0, h2d.T)
            vmax = max(1, np.max(h2d))
            im = ax.pcolormesh(satz_bins, diff_bins, masked,
                               cmap='plasma',
                               norm=LogNorm(vmin=1, vmax=vmax))

            # Satzen range boundaries and labels
            for lo, hi, lbl in config.SATZEN_RANGES:
                if lo > satz_bins[0]:
                    ax.axvline(lo, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
                mid = (lo + hi) / 2.0
                ax.text(mid, 1.02, lbl,
                        transform=ax.get_xaxis_transform(),
                        ha='center', va='bottom', fontsize=7, style='italic')

            plt.colorbar(im, ax=ax, label='Count (log scale)')
            ax.set_xlabel('Satellite Zenith Angle (°)')
            ax.set_ylabel(var_titles[dv])
            ax.set_title(cls_key)
            if dv in config.XLIMS:
                ax.set_ylim(config.XLIMS[dv])

        fig.suptitle(f"{title_prefix}Satzen vs {var_titles[dv]}", fontsize=12, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig)
        if png_dir:
            fname = f"{date_tag}_{dv}.png" if date_tag else f"{dv}.png"
            fig.savefig(os.path.join(png_dir, fname), dpi=300, bbox_inches='tight')
        plt.close(fig)


def generate_pdf_report(meta, master_counts, master_counts_miz=None,
                        satzen_hist=None, satzen_2d=None,
                        satzen_hist_miz=None, satzen_2d_miz=None):
    # All output lives inside a folder named after the PDF
    report_name = os.path.splitext(config.PDF_FILENAME)[0]
    report_dir  = os.path.join(config.OUTPUT_DIR, report_name)

    dirs = {
        'metadata':            os.path.join(report_dir, '01_metadata'),
        'histograms':          os.path.join(report_dir, '02_histograms'),
        'miz_histograms':      os.path.join(report_dir, '03_miz_histograms'),
        'satzen_histograms':   os.path.join(report_dir, '04_satzen_histograms'),
        'satzen_scatter':      os.path.join(report_dir, '05_satzen_scatter'),
        'miz_satzen_hist':     os.path.join(report_dir, '06_miz_satzen_histograms'),
        'miz_satzen_scatter':  os.path.join(report_dir, '07_miz_satzen_scatter'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    pdf_path = os.path.join(report_dir, config.PDF_FILENAME)
    print(f"Generating Report: {pdf_path}")
    
    with PdfPages(pdf_path) as pdf:
        # PAGE 1: METADATA
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

        satzen_str = "Disabled"
        if config.ENABLE_SATZEN_ANALYSIS:
            ranges_str = ", ".join(f"{lo}–{hi}°" for lo, hi, _ in config.SATZEN_RANGES)
            satzen_str = f"Active | Mode: {config.SATZEN_PLOT_MODE} | Ranges: {ranges_str}"

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
            f"Region:           {config.SELECTED_REGION.upper()}\n"
            f"Bounding Box:     Lat {config.CROP_BOUNDS['lat_min']} to {config.CROP_BOUNDS['lat_max']}\n"
            f"                  Lon {config.CROP_BOUNDS['lon_min']} to {config.CROP_BOUNDS['lon_max']}\n"
            f"Max Solar Zenith: {config.MAX_SOLAR_ZENITH}°\n\n"
            f"MIZ SUBSET CONFIGURATION\n"
            f"------------------------------------------------------------\n"
            f"Filtering:        {miz_str}\n\n"
            f"SATZEN ANALYSIS\n"
            f"------------------------------------------------------------\n"
            f"Configuration:    {satzen_str}\n\n"
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
            cmap='plasma', zorder=2, alpha=0.8,
            rasterized=True
        )
        
        bounds = config.CROP_BOUNDS
        
        l1_x = np.linspace(bounds['lon_min'], bounds['lon_max'], 100)
        l1_y = np.full(100, bounds['lat_min'])
        l2_x = np.full(100, bounds['lon_max'])
        l2_y = np.linspace(bounds['lat_min'], bounds['lat_max'], 100)
        l3_x = np.linspace(bounds['lon_max'], bounds['lon_min'], 100)
        l3_y = np.full(100, bounds['lat_max'])
        l4_x = np.full(100, bounds['lon_min'])
        l4_y = np.linspace(bounds['lat_max'], bounds['lat_min'], 100)
        
        box_lons = np.concatenate([l1_x, l2_x, l3_x, l4_x])
        box_lats = np.concatenate([l1_y, l2_y, l3_y, l4_y])
        
        ax.plot(box_lons, box_lats, color='black', linewidth=2, 
                transform=ccrs.PlateCarree(), label='Analysis Region', zorder=10)
        
        cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label('Scene Overlap Count')
        ax.set_title("Geographic Coverage Density")
        
        date_range_str = _format_date_range(meta['time_start'], meta['time_end'])
        date_pfx = f"{date_range_str}\n"
        date_tag = date_range_str.replace(' – ', '_to_')  # e.g. 2025-08-01_to_2025-08-14

        pdf.savefig()
        fig.savefig(os.path.join(dirs['metadata'], f"{date_tag}_metadata.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # PART 1: Standard Plots
        _plot_histogram_pages(pdf, master_counts, title_prefix=date_pfx,
                              png_dir=dirs['histograms'], date_tag=date_tag)

        # PART 2: MIZ Subset Plots
        if config.ENABLE_MIZ_HISTOGRAMS and master_counts_miz:
            plt.figure(figsize=(11, 8.5))
            plt.text(0.5, 0.5,
                     f"MIZ SUBSET ANALYSIS\n\nTemperature Filter: {config.MIZ_T11_RANGE[0]}K < T11 < {config.MIZ_T11_RANGE[1]}K",
                     ha='center', va='center', fontsize=20)
            plt.axis('off')
            pdf.savefig()
            plt.close()

            _plot_histogram_pages(pdf, master_counts_miz, title_prefix=f"[MIZ Subset] {date_pfx}",
                                  png_dir=dirs['miz_histograms'], date_tag=date_tag)

        # PART 3: Satzen Analysis
        if config.ENABLE_SATZEN_ANALYSIS and (satzen_hist is not None or satzen_2d is not None):
            plt.figure(figsize=(11, 8.5))
            plt.text(0.5, 0.5,
                     "SATELLITE ZENITH ANGLE ANALYSIS\n\n"
                     "Brightness temperature differences for IN and WN\n"
                     "split by three satzen ranges",
                     ha='center', va='center', fontsize=18)
            plt.axis('off')
            pdf.savefig()
            plt.close()

            mode = config.SATZEN_PLOT_MODE
            if mode in ('histogram', 'both') and satzen_hist is not None:
                _plot_satzen_histogram_pages(pdf, satzen_hist, title_prefix=date_pfx,
                                             png_dir=dirs['satzen_histograms'], date_tag=date_tag)
            if mode in ('scatter', 'both') and satzen_2d is not None:
                _plot_satzen_scatter_pages(pdf, satzen_2d, title_prefix=date_pfx,
                                           png_dir=dirs['satzen_scatter'], date_tag=date_tag)

            # MIZ subset
            if (config.ENABLE_MIZ_HISTOGRAMS
                    and (satzen_hist_miz is not None or satzen_2d_miz is not None)):
                plt.figure(figsize=(11, 8.5))
                plt.text(0.5, 0.5,
                         f"SATELLITE ZENITH ANGLE ANALYSIS — MIZ SUBSET\n\n"
                         f"Temperature filter: "
                         f"{config.MIZ_T11_RANGE[0]}K < T11 < {config.MIZ_T11_RANGE[1]}K",
                         ha='center', va='center', fontsize=18)
                plt.axis('off')
                pdf.savefig()
                plt.close()

                if mode in ('histogram', 'both') and satzen_hist_miz is not None:
                    _plot_satzen_histogram_pages(pdf, satzen_hist_miz, title_prefix=f"[MIZ] {date_pfx}",
                                                 png_dir=dirs['miz_satzen_hist'], date_tag=date_tag)
                if mode in ('scatter', 'both') and satzen_2d_miz is not None:
                    _plot_satzen_scatter_pages(pdf, satzen_2d_miz, title_prefix=f"[MIZ] {date_pfx}",
                                               png_dir=dirs['miz_satzen_scatter'], date_tag=date_tag)