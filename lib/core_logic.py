import numpy as np
from scipy import ndimage
from . import config

def compute_neighbors(classification, cloud_bool):
    struct3 = ndimage.generate_binary_structure(2, 1)
    
    ice_bool = (classification == 2)
    water_bool = (classification == 1)

    # --- Ice Neighbors (near Water) ---
    dilated_water = ndimage.binary_dilation(water_bool, structure=struct3)
    iNw = dilated_water & ice_bool & (~water_bool)
    i2Nw = ndimage.binary_dilation(iNw, structure=struct3) & ice_bool & (~iNw)
    i3Nw = ndimage.binary_dilation(i2Nw, structure=struct3) & ice_bool & (~iNw) & (~i2Nw)

    # --- Water Neighbors (near Ice) ---
    dilated_ice = ndimage.binary_dilation(ice_bool, structure=struct3)
    wNi = dilated_ice & water_bool & (~ice_bool)
    w2Ni = ndimage.binary_dilation(wNi, structure=struct3) & water_bool & (~wNi)
    w3Ni = ndimage.binary_dilation(w2Ni, structure=struct3) & water_bool & (~wNi) & (~w2Ni)

    # --- Cloud Neighbors (Edges against Ice/Water) ---
    iNc = ndimage.binary_dilation(cloud_bool, structure=struct3) & ice_bool
    i2Nc = ndimage.binary_dilation(iNc, structure=struct3) & ice_bool & (~iNc)
    wNc = ndimage.binary_dilation(cloud_bool, structure=struct3) & water_bool
    w2Nc = ndimage.binary_dilation(wNc, structure=struct3) & water_bool & (~wNc)

    masks = {
        'iNw': iNw, 'i2Nw': i2Nw, 'i3Nw': i3Nw,
        'wNi': wNi, 'w2Ni': w2Ni, 'w3Ni': w3Ni,
        'iNc': iNc, 'i2Nc': i2Nc, 'wNc': wNc, 'w2Nc': w2Nc
    }

    # --- Mixed Logic (Intersections) ---
    # Sum up how many edge types a pixel belongs to
    neighbor_sum = np.zeros_like(classification, dtype=int)
    for k in masks:
        neighbor_sum += masks[k].astype(int)
        
    masks['Mixed'] = (neighbor_sum >= 2)

    # --- Interior / "No Neighbor" Logic ---
    # Pixels that are valid class but NOT an edge and NOT mixed
    # Since Mixed implies at least 2 edges, checking neighbor_sum == 0 is sufficient
    
    masks['IN'] = ice_bool & (neighbor_sum == 0)
    masks['WN'] = water_bool & (neighbor_sum == 0)

    # --- Interior Cloud (CN) ---
    # A cloud pixel is "Interior" if it is not touching Ice or Water.
    # Note: 'dilated_ice' encompasses Ice + its immediate boundary.
    # If a cloud pixel is True, and dilated_ice is True, that cloud is touching ice.
    
    cloud_touching_ice = cloud_bool & dilated_ice
    cloud_touching_water = cloud_bool & dilated_water
    
    # CN = Cloud AND NOT touching ice AND NOT touching water
    masks['CN'] = cloud_bool & (~cloud_touching_ice) & (~cloud_touching_water)

    return masks

def create_classification(cma, landuse, lat, sunz):
    # 1. Day Mask
    day_mask = (sunz <= config.MAX_SOLAR_ZENITH)
    
    # 2. Geography
    is_water_geo = (landuse == 16)
    is_land_geo  = (landuse != 16)

    # 3. Clouds (Remove clouds over land)
    cloud_raw = ((cma == 1) | (cma == 2))
    cloud_clean = cloud_raw.copy()
    cloud_clean[is_land_geo] = False 

    # 4. Classification
    cls = np.zeros_like(cma, dtype=np.uint8)
    cls[is_water_geo] = 1
    cls[is_land_geo] = 4
    cls[is_water_geo & (cma == 3)] = 2
    cls[cloud_clean] = 3
    
    # 5. Cleanup
    cls[~day_mask] = 0
    cls[cma == 255] = 0
    cls[is_land_geo] = 4 # Ensure land overrides
    
    return cls, cloud_clean