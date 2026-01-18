import os
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from scipy import ndimage
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerPatch

# path for data
data_dir = '/home/oliver/Documents/MET/VIIRS_Mask/data-case1'
output_dir = '/home/oliver/Documents/MET/VIIRS_Mask/output_mixed_NB'

# load files 
nc_files = [f for f in os.listdir(data_dir) if f.endswith('.nc')]
cloud_ice_file = None
land_water_file = None
for f in nc_files:
    if 'CMA_noaa21' in f:
        cloud_ice_file = os.path.join(data_dir, f)
    elif 'physiography_noaa21' in f:
        land_water_file = os.path.join(data_dir, f)

ds_cloud = Dataset(cloud_ice_file, 'r')
ds_land = Dataset(land_water_file, 'r')

cloud_data = ds_cloud.variables['cma_extended'][0, :, :]
land_data  = ds_land.variables['landuse'][0, :, :]

# Subset 
subset = True
if subset:
    x_min, x_max = 2005, 2286
    y_min, y_max = 610, 749
    cloud_data = cloud_data[y_min:y_max+1, x_min:x_max+1]
    land_data  = land_data[y_min:y_max+1, x_min:x_max+1]

cloud_data = np.array(cloud_data)
land_data  = np.array(land_data)

# No-data mask
no_data_mask = (cloud_data == 255) | (land_data == 255)

# Basic masks
cloudfree_mask  = (cloud_data == 0)
cloud_mask_raw  = ((cloud_data == 1) | (cloud_data == 2))
cfree_land_mask = ((land_data != 16) & cloudfree_mask)
cfree_water_mask= ((land_data == 16) & cloudfree_mask)
all_land_mask   = (land_data != 16)
ice_mask_raw    = (cloud_data == 3)

# Remove clouds over land: "keep it just green"
# cloud_mask should NOT include clouds that lie over land pixels
# (so land remains green)
cloud_mask = cloud_mask_raw.copy()
cloud_mask[all_land_mask] = False

# Choose land & water masks (same as you had)
land_mask  = all_land_mask
water_mask = cfree_water_mask

# Structuring element
struct3 = ndimage.generate_binary_structure(2, 1)

# Colors and labels (kept same)
colors = [
    'black',    # 0 No data
    'blue',     # 1 Water
    'white',    # 2 Ice
    'gray',     # 3 Clouds
    'green',    # 4 Land
    'red',      # 5 iNw
    'orange',   # 6 iNNw
    'gold',     # 7 iNNNw
    'cyan',     # 8 wNi
    'navy',     # 9 wNNi
    'purple'    # 10 wNNNi
]

labels = [
    'No data',
    'Water',
    'Ice',
    'Clouds',
    'Land',
    'InW',
    'I2nW',
    'I3nW',
    'WnI',
    'W2nI',
    'W3nI'
]

cmap = ListedColormap(colors)

# Base classification (unchanged)
classification_base = np.zeros(cloud_data.shape, dtype=np.uint8)
classification_base[no_data_mask] = 0
classification_base[cfree_water_mask] = 1
classification_base[ice_mask_raw] = 2
classification_base[cloud_mask] = 3
classification_base[land_mask] = 4

# helper: is ice?
def is_ice_color(pixel_val):
    return pixel_val == 2

colors_dict = {
    'water': 'blue',
    'ice'  : 'white',
    'cloud': 'gray',
    'land' : 'green'
}

# Legend handler (keeps your previous combined handler if needed)
class HalfPatchHandler(HandlerPatch):
    def __init__(self, left_color, center_color, right_color):
        self.left_color = left_color
        self.center_color = center_color
        self.right_color = right_color
        super().__init__()

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        p1 = Rectangle([xdescent, ydescent], width/3, height,
                       facecolor=self.left_color, hatch='///',
                       edgecolor='gray', transform=trans)
        p2 = Rectangle([xdescent + width/3, ydescent], width/3, height,
                       facecolor=self.center_color, hatch='///',
                       edgecolor='gray', transform=trans)
        p3 = Rectangle([xdescent + 2*width/3, ydescent], width/3, height,
                       facecolor=self.right_color, hatch='///',
                       edgecolor='gray', transform=trans)
        return [p1, p2, p3]

# Helper to compute all neighbor masks (so we can reuse them)
# Returns dict of masks: iNw,i2Nw,i3Nw,wNi,w2Ni,w3Ni,iNc,i3Nc,wNc,w2Nc
def compute_all_neighbors(classification):
    """Compute first/second/third neighbors for ice<->water and cloud neighbors.
       Input classification is a base classification array where:
         water == 1, ice == 2, cloud == 3, land == 4, no-data == 0
    """
    ice_bool  = (classification == 2)
    water_bool= (classification == 1)
    cloud_bool= cloud_mask.astype(bool)   # use the cleaned cloud_mask (no clouds over land)

    # Ice-side neighbors (iNw, i2Nw, i3Nw) - ice pixels that are near water
    dilated_water = ndimage.binary_dilation(water_bool, structure=struct3)
    iNw   = dilated_water & ice_bool & (~water_bool)
    i2Nw  = ndimage.binary_dilation(iNw, structure=struct3) & ice_bool & (~iNw)
    i3Nw = ndimage.binary_dilation(i2Nw, structure=struct3) & ice_bool & (~iNw) & (~i2Nw)

    # Water-side neighbors (wNi, w2Ni, w3Ni) - water pixels that are near ice
    dilated_ice = ndimage.binary_dilation(ice_bool, structure=struct3)
    wNi   = dilated_ice & water_bool & (~ice_bool)
    w2Ni  = ndimage.binary_dilation(wNi, structure=struct3) & water_bool & (~wNi)
    w3Ni = ndimage.binary_dilation(w2Ni, structure=struct3) & water_bool & (~wNi) & (~w2Ni)

    # Cloud neighbors: pixels of ice/water that are near cloud
    iNc  = ndimage.binary_dilation(cloud_bool, structure=struct3) & ice_bool
    i2Nc = ndimage.binary_dilation(iNc, structure=struct3) & ice_bool & (~iNc)

    wNc  = ndimage.binary_dilation(cloud_bool, structure=struct3) & water_bool
    w2Nc = ndimage.binary_dilation(wNc, structure=struct3) & water_bool & (~wNc)

    return {
        'iNw': iNw, 'i2Nw': i2Nw, 'i3Nw': i3Nw,
        'wNi': wNi, 'w2Ni': w2Ni, 'w3Ni': w3Ni,
        'iNc': iNc, 'i2Nc': i2Nc, 'wNc': wNc, 'w2Nc': w2Nc
    }

# Mixed neighbors function (stack method)
# Input: dict of neighbor masks as above

#def compute_mixed_neighbors_from_masks(neighbor_masks):
   # all_neighbors = [
      #  neighbor_masks['iNw'], neighbor_masks['i2Nw'],
      #  neighbor_masks['wNi'], neighbor_masks['w2Ni'],
       # neighbor_masks['iNc'], neighbor_masks['i2Nc'],
       # neighbor_masks['wNc'], neighbor_masks['w2Nc']
   # ]
   # neighbor_stack = np.stack(all_neighbors, axis=0)
   # mixed = (np.sum(neighbor_stack, axis=0) >= 2)
    #return mixed



# Compute all neighbors 
neighbor_masks_all = compute_all_neighbors(classification_base)

#  Mixed neighbors without np.stack 
# Sum booleans directly (convert to int)
neighbor_sum = (
    neighbor_masks_all['iNw'].astype(int) +
    neighbor_masks_all['i2Nw'].astype(int) +
    neighbor_masks_all['wNi'].astype(int) +
    neighbor_masks_all['w2Ni'].astype(int) +
    neighbor_masks_all['iNc'].astype(int) +
    neighbor_masks_all['i2Nc'].astype(int) +
    neighbor_masks_all['wNc'].astype(int) +
    neighbor_masks_all['w2Nc'].astype(int)
)

# Mixed neighbors: pixels belonging to ≥ 2 neighbor types
mixed_all = (neighbor_sum >= 2)




# Plot helper 
#   - classification: classification array (with neighbor labels if desired)
#   - neighbor_masks_to_draw: dict of masks to draw (keys) -> value True means draw
#   - title & save_path optional

# plot_case with cloud neighbors in light gray, mixed neighbors in orchid
def plot_case(classification, neighbor_masks, cloud_neighbor_masks, mixed_mask, title, filename):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(classification, interpolation='none', cmap=cmap, vmin=0, vmax=10)
    ax.set_title(title)
    ax.axis('off')

    # Draw neighbors (ice->water & water->ice) in their colors
    for name, mask in neighbor_masks.items():
        if name == 'iNw':
            disp_color = colors[5]
        elif name == 'i2Nw':
            disp_color = colors[6]
        elif name == 'i3Nw':
            disp_color = colors[7]
        elif name == 'wNi':
            disp_color = colors[8]
        elif name == 'w2Ni':
            disp_color = colors[9]
        elif name == 'w3Ni':
            disp_color = colors[10]
        else:
            disp_color = 'magenta'
        for (r, c), val in np.ndenumerate(mask):
            if val:
                ax.add_patch(mpatches.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    facecolor=disp_color, edgecolor='none'
                ))


     # Draw cloud neighbors with two tones of gray 
    for name, mask in cloud_neighbor_masks.items():
        if name in ['iNc', 'wNc']:
            disp_color = 'lightgray'   # 1st cloud neighbors
        elif name in ['i2Nc', 'w2Nc']:
            disp_color = 'lightslategray'   # 2nd cloud neighbors
        else:
            disp_color = 'lightgray'
        for (r, c), val in np.ndenumerate(mask):
            if val:
                ax.add_patch(mpatches.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    facecolor=disp_color, edgecolor='none'
                ))

    # Draw mixed neighbors in orchid
    if mixed_mask is not None:
        for (r, c), val in np.ndenumerate(mixed_mask):
            if val:
                ax.add_patch(mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                               facecolor='orchid', edgecolor='black', linewidth=0))

    # Build legend dynamically (only show what appears)
    unique_vals = set(classification.flatten())
    patches = []
    
    # Base classification colors
    for val in unique_vals:
        if val < len(colors):
            patches.append(mpatches.Patch(color=colors[val], label=labels[val]))
    
    # Ice ↔ Water Neighbors 
    neighbor_labels_colors = {
        'iNw':    ('InW', colors[5]),
        'i2Nw':   ('I2nW', colors[6]),
        'i3Nw':  ('I3nW', colors[7]),
        'wNi':    ('WnI', colors[8]),
        'w2Ni':   ('W2nI', colors[9]),
        'w3Ni':  ('W3nI', colors[10]),
    }
    
    for key, (lab, col) in neighbor_labels_colors.items():
        if key in neighbor_masks and np.any(neighbor_masks[key]):
            patches.append(mpatches.Patch(color=col, label=lab))
    
    #  Cloud Neighbors (light and darker gray) 
    if cloud_neighbor_masks and any(np.any(m) for m in cloud_neighbor_masks.values()):
        if any(np.any(cloud_neighbor_masks[k]) for k in ['iNc', 'wNc']):
            patches.append(mpatches.Patch(color='lightgray', label='1st Cloud NB'))
        if any(np.any(cloud_neighbor_masks[k]) for k in ['i2Nc', 'w2Nc']):
            patches.append(mpatches.Patch(color='lightslategray', label='2nd Cloud NB'))
    
    #  Mixed Neighbors (orchid) 
    if mixed_mask is not None and np.any(mixed_mask):
        patches.append(mpatches.Patch(color='orchid', label='Mixed neighbors'))
    
    ax.legend(handles=patches, loc='lower right', fontsize='small', frameon=True)


    ax.legend(handles=patches, loc='lower right', fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))


# Precompute all neighbor masks
#neighbor_masks_all = compute_all_neighbors(classification_base)
#mixed_all = compute_mixed_neighbors_from_masks(neighbor_masks_all)


# Just the map
plot_case(
    classification_base,
    neighbor_masks={},
    cloud_neighbor_masks={},
    mixed_mask=None,
    title='Classification Map',
    filename = "classification_map.png",
)
# 1) Only classification with ice/water neighbors
plot_case(
    classification_base,
    neighbor_masks={k: neighbor_masks_all[k] for k in ['iNw','i2Nw','i3Nw','wNi','w2Ni','w3Ni']},
    cloud_neighbor_masks={},
    mixed_mask=None,
    title='Classification with Ice/Water Neighbors',
    filename = "only_NB.png",
    )


# 2) Classification with all neighbor types (ice, water, cloud, mixed)
plot_case(
    classification_base,
    neighbor_masks={},
    cloud_neighbor_masks={k: neighbor_masks_all[k] for k in ['iNc','i2Nc','wNc','w2Nc']},
    mixed_mask=mixed_all,
    title='Cloud neighbors with Mixed',
    filename = "cloud_NB.png",
   )

# 3) Ice neighbors + mixed
plot_case(classification_base,
                  neighbor_masks={k: neighbor_masks_all[k] for k in ['iNw','i2Nw','i3Nw']},
                  cloud_neighbor_masks={},
                  mixed_mask=mixed_all,
                  title='Ice neighbors with Mixed',
                  filename = "ice_and_mixed_NB.png",
   )


# 4) Water neighbors + mixed
plot_case(classification_base,
                  neighbor_masks={k: neighbor_masks_all[k] for k in ['wNi','w2Ni','w3Ni']},
                  cloud_neighbor_masks={},
                  mixed_mask=mixed_all,
                  title='Water neighbors with Mixed',
                  filename = "water_and_mixed_NB.png",
           )


# 5) Only mixed neighbors
plot_case(classification_base,
                  neighbor_masks={},
                  cloud_neighbor_masks={},
                  mixed_mask=mixed_all,
                  title='Only Mixed Neighbors',
                  filename = "only_mixed_NB.png",
           )


# Histograms for each neighbor group
for f in nc_files:
    if 'S_NWC_viirs_noaa21' in f:
        sat_data = os.path.join(data_dir, f)

ds_sat = Dataset(sat_data, 'r')

        
tb_11 = ds_sat.variables['image3'][0, :, :]
tb_11 = tb_11[y_min:y_max+1, x_min:x_max+1]


tb_12 = ds_sat.variables['image4'][0, :, :]
tb_12 = tb_12[y_min:y_max+1, x_min:x_max+1]

tb_37 = ds_sat.variables['image5'][0, :, :]
tb_37 = tb_37[y_min:y_max+1, x_min:x_max+1]

tb_87 = ds_sat.variables['image7'][0, :, :]
tb_87 = tb_87[y_min:y_max+1, x_min:x_max+1]


# Define masks per neighbor type
masks_cloud = {
    "InC": neighbor_masks_all['iNc'],
    "I2nC": neighbor_masks_all['i2Nc'],
    "WnC": neighbor_masks_all['wNc'],
    "W2nC": neighbor_masks_all['w2Nc'],
}

masks_ice = {
    "InW": neighbor_masks_all['iNw'],
    "I2nW": neighbor_masks_all['i2Nw'],
    "I3nW": neighbor_masks_all['i3Nw'],
}

masks_water = {
    "WnI": neighbor_masks_all['wNi'],
    "W2nI": neighbor_masks_all['w2Ni'],
    "W3nI": neighbor_masks_all['w3Ni'],
}

masks_mixed = {
    "Mixed": mixed_all
}



def plot_histograms(data, masks, title, xlabel,filename, xlim=None):
    plt.figure(figsize=(10, 6))

    for name, mask in masks.items():
        values = data[mask]
        if values.size > 0:
            counts, bin_edges = np.histogram(values, bins=50, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            plt.plot(bin_centers, counts, label=name, linewidth=2)

    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.title(title)
    if xlim:
        plt.xlim(xlim)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()




#  Case 1: Cloud neighbors 
plot_histograms(tb_11, masks_cloud,
                title="Histogram of T11 (Cloud Neighbors)",
                xlabel="T11 (K)",
                filename = "hist_tb11_cloud.png",
                xlim = (255,275))

#  Case 2: Ice neighbors 
plot_histograms(tb_11, masks_ice,
                title="Histogram of T11 (Ice Neighbors)",
                xlabel="T11 (K)",
                filename = "hist_tb11_ice.png",
                xlim = (255,275))

#  Case 3: Water neighbors 
plot_histograms(tb_11, masks_water,
                title="Histogram of T11 (Water Neighbors)",
                xlabel="T11 (K)",
                filename = "hist_tb11_water.png",
                xlim = (255,275))

#  Case 4: Mixed neighbors 
plot_histograms(tb_11, masks_mixed,
                title="Histogram of T11 (Mixed Neighbors)",
                xlabel="T11 (K)",
                filename = "hist_tb11_mixed.png",
                xlim = (255,275))

# Differences
tb_diff1 = tb_11 - tb_12
tb_diff2 = tb_37 - tb_12
tb_diff3 = tb_87 - tb_12  


# Case 3: T11 - T12 
plot_histograms(tb_diff1, masks_ice,
                title="Histogram of T11 - T12 (Ice Neighbors)",
                xlabel="T11 - T12 (K)",
                filename = "hist_tb11_tb12_ice.png",
                xlim = (0,2))

plot_histograms(tb_diff1, masks_water,
                title="Histogram of T11 - T12 (Water Neighbors)",
                xlabel="T11 - T12 (K)",
                filename = "hist_tb11_tb12_water.png",
                xlim = (0,2))

plot_histograms(tb_diff1,masks_mixed,
                title = "Histogram of T11 - T12 (Mixed Neighbors)",
                xlabel="T11 - T12 (K)",
                filename = "hist_tb11_tb12_mixed.png",
                xlim = (0,2))

plot_histograms(tb_diff1, masks_cloud,
                title="Histogram of T11 - T12 (Cloud Neighbors)",
                xlabel="T11 - T12 (K)",
                filename ="hist_tb11_tb12_cloud.png",
                xlim = (0,2))


# Case 4: T37 - T12 
plot_histograms(tb_diff2, masks_ice,
                title="Histogram of T37 - T12 (Ice Neighbors)",
                xlabel="T37 - T12 (K)",
                filename = "hist_tb37_tb12_ice.png",
                xlim = (0,11))

plot_histograms(tb_diff2, masks_water,
                title="Histogram of T37 - T12 (Water Neighbors)",
                xlabel="T37 - T12 (K)",
                filename = "hist_tb37_tb12_water.png",
                xlim = (0,11))

plot_histograms(tb_diff2,masks_mixed,
                title = "Histogram of T37 - T12 (Mixed Neighbors)",
                xlabel="T37 - T12 (K)",
                filename = "hist_tb37_tb12_mixed.png",
                xlim = (0,11))

plot_histograms(tb_diff2, masks_cloud,
                title="Histogram of T37 - T12 (Cloud Neighbors)",
                xlabel="T37 - T12 (K)",
                filename = "hist_tb37_tb12_cloud.png",
                xlim = (0,11))

# Case 5: T8.7 - T12
plot_histograms(tb_diff3, masks_ice,
                title="Histogram of T8.7 - T12 (Ice Neighbors)",
                xlabel="T8.7 - T12 (K)",
                filename = "hist_tb87_tb12_ice.png",
                xlim = (-2.5,1))

plot_histograms(tb_diff3, masks_water,
                title="Histogram of T8.7 - T12 (Water Neighbors)",
                xlabel="T8.7 - T12 (K)",
                filename = "hist_tb87_tb12_water.png",
                xlim = (-2.5,1))

plot_histograms(tb_diff3,masks_mixed,
                title = "Histogram of T8.7 - T12 (Mixed Neighbors)",
                xlabel="T8.7 - T12 (K)",
                filename = "hist_tb87_tb12_mixed.png",
                xlim = (-2.5,1))

plot_histograms(tb_diff3, masks_cloud,
                title="Histogram of T8.7 - T12 (Cloud Neighbors)",
                xlabel="T8.7 - T12 (K)",
                filename = "hist_tb87_tb12_cloud.png",
                xlim = (-2.5,1))

#  open textures dataset 
for f in nc_files:
    if 'S_NWC_textures_noaa21' in f:
        tex_data = os.path.join(data_dir, f)
ds_tex = Dataset(tex_data, 'r')

#  Variables from S_NWC_viirs 
sunzenith = ds_sat.variables['sunzenith'][0, :, :]
sunzenith =sunzenith[y_min:y_max+1, x_min:x_max+1]
satzenith = ds_sat.variables['satzenith'][0, :, :]
satzenith = satzenith[y_min:y_max+1, x_min:x_max+1]

#  Variables from S_NWC_textures (apply scaling) 
r06_tex    = ds_tex.variables['r06'][0, :, :].astype(float)
r06_tex=r06_tex[y_min:y_max+1, x_min:x_max+1]

t11_tex    = ds_tex.variables['t11'][0, :, :].astype(float)
t11_tex=t11_tex[y_min:y_max+1, x_min:x_max+1]

t11t12_tex = ds_tex.variables['t11t12'][0, :, :].astype(float)
t11t12_tex=t11t12_tex[y_min:y_max+1, x_min:x_max+1]

t37t12_tex = ds_tex.variables['t37t12'][0, :, :].astype(float)
t37t12_tex=t37t12_tex[y_min:y_max+1, x_min:x_max+1]



#  Plot zeniths 
plot_histograms(sunzenith, masks_ice,
                title="Histogram of Sun Zenith (Ice Neighbors)",
                xlabel="Sun Zenith (degrees)",
                filename = "hist_sunzenith_ice.png")

plot_histograms(sunzenith, masks_water,
                title="Histogram of Sun Zenith (Water Neighbors)",
                xlabel="Sun Zenith (degrees)",
                filename = "hist_sunzenith_water.png")

plot_histograms(sunzenith,masks_mixed,
                title= "Hstogram of Sun Zenith (Mixed Neighbors)",
                xlabel="Sun Zenith (degrees)",
                filename = "hist_sunzenith_mixed.png")

plot_histograms(sunzenith, masks_cloud,
                title="Histogram of Sun Zenith  (Cloud Neighbors)",
                xlabel="Sun Zenith (degrees)",
                filename = "hist_sunzenith_cloud.png")

plot_histograms(satzenith, masks_ice,
                title="Histogram of Sat Zenith (Ice Neighbors)",
                xlabel="Sat Zenith (degrees)",
                filename = "hist_satzenith_ice.png")

plot_histograms(satzenith, masks_water,
                title="Histogram of Sat Zenith (Water Neighbors)",
                xlabel="Sat Zenith (degrees)",
                filename = "hist_satzenith_water.png")

plot_histograms(satzenith,masks_mixed,
                title= "Hstogram of Sat Zenith (Mixed Neighbors)",
                xlabel="Sat Zenith (degrees)",
                filename = "hist_satzenith_mixed.png")

plot_histograms(satzenith, masks_cloud,
                title="Histogram of Sat Zenith  (Cloud Neighbors)",
                xlabel="Sat Zenith (degrees)",
                filename = "hist_satzenith_cloud.png")

#  Plot textures 
plot_histograms(r06_tex, masks_ice,
                title="Histogram of r06-texture (Ice Neighbors)",
                xlabel="r06-texture ",
                filename = "hist_r06_texture_ice.png")

plot_histograms(r06_tex, masks_water,
                title="Histogram of r06-texture (Water Neighbors)",
                xlabel="r06-texture ",
                filename = "hist_r06_texture_water.png")

plot_histograms(r06_tex,masks_mixed,
                title= "Hstogram of r06_texture (Mixed Neighbors)",
                xlabel="r06_texture",
                filename = "hist_r06_texture_mixed.png")

plot_histograms(r06_tex, masks_cloud,
                title="Histogram of r06_texture (Cloud Neighbors)",
                xlabel="r06_texture",
                filename = "hist_r06_texture_cloud.png")

plot_histograms(t11_tex, masks_ice,
                title="Histogram of t11-texture (Ice Neighbors)",
                xlabel="t11-texture ",
                filename = "hist_t11_texture_ice.png")

plot_histograms(t11_tex, masks_water,
                title="Histogram of t11-texture (Water Neighbors)",
                xlabel="t11-texture",
                filename = "hist_t11_texture_water.png")

plot_histograms(t11_tex,masks_mixed,
                title= "Hstogram of t11_texture (Mixed Neighbors)",
                xlabel="t11_texture",
                filename = "hist_t11_texture_mixed.png")

plot_histograms(t11_tex, masks_cloud,
                title="Histogram of t11_texture (Cloud Neighbors)",
                xlabel="t11_texture",
                filename = "hist_t11_texture_cloud.png")


plot_histograms(t11t12_tex, masks_ice,
                title="Histogram of t11t12-texture (Ice Neighbors)",
                xlabel="t11t12-texture",
                filename = "hist_t11t12_texture_ice.png",
                xlim = (0,0.3))

plot_histograms(t11t12_tex, masks_water,
                title="Histogram of t11t12-texture (Water Neighbors)",
                xlabel="t11t12-texture ",
                filename = "hist_t11t12_texture_water.png",
                xlim = (0,0.3))

plot_histograms(t11t12_tex,masks_mixed,
                title= "Hstogram of t11t12_texture (Mixed Neighbors)",
                xlabel="t11t12_texture",
                filename = "hist_t11t12_texture_mixed.png")

plot_histograms(t11t12_tex, masks_cloud,
                title="Histogram of t11t12_texture (Cloud Neighbors)",
                xlabel="t11t12_texture",
                filename = "hist_t11t12_texture_cloud.png")

plot_histograms(t37t12_tex, masks_ice,
                title="Histogram of t37t12-texture (Ice Neighbors)",
                xlabel="t37t12-texture ",
                filename = "hist_t37t12_texture_ice.png",
                xlim = (0,6))

plot_histograms(t37t12_tex, masks_water,
                title="Histogram of t37t12-texture (Water Neighbors)",
                xlabel="t37t12-texture ",
                filename = "hist_t37t12_texture_water.png",
                xlim = (0,4))

plot_histograms(t37t12_tex,masks_mixed,
                title= "Hstogram of t37t12_texture (Mixed Neighbors)",
                xlabel="t37t12_texture",
                filename = "hist_t37t12_texture_mixed.png",
                xlim = (0,4))

plot_histograms(t37t12_tex, masks_cloud,
                title="Histogram of t37t12_texture (Cloud Neighbors)",
                xlabel="t37t12_texture",
                filename = "hist_t37t12_texture_cloud.png")
