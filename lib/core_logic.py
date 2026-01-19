import numpy as np
from scipy import ndimage

def compute_neighbors(classification, cloud_bool):
    """
    Computes boolean masks for 1st, 2nd, and 3rd neighbors.
    classification: 2D array where 1=Water, 2=Ice
    cloud_bool: 2D boolean array where True=Cloud
    """
    struct3 = ndimage.generate_binary_structure(2, 1)
    ice_bool = (classification == 2)
    water_bool = (classification == 1)

    # --- Ice Neighbors ---
    dilated_water = ndimage.binary_dilation(water_bool, structure=struct3)
    iNw = dilated_water & ice_bool & (~water_bool)
    i2Nw = ndimage.binary_dilation(iNw, structure=struct3) & ice_bool & (~iNw)
    i3Nw = ndimage.binary_dilation(i2Nw, structure=struct3) & ice_bool & (~iNw) & (~i2Nw)

    # --- Water Neighbors ---
    dilated_ice = ndimage.binary_dilation(ice_bool, structure=struct3)
    wNi = dilated_ice & water_bool & (~ice_bool)
    w2Ni = ndimage.binary_dilation(wNi, structure=struct3) & water_bool & (~wNi)
    w3Ni = ndimage.binary_dilation(w2Ni, structure=struct3) & water_bool & (~wNi) & (~w2Ni)

    # --- Cloud Neighbors ---
    iNc = ndimage.binary_dilation(cloud_bool, structure=struct3) & ice_bool
    i2Nc = ndimage.binary_dilation(iNc, structure=struct3) & ice_bool & (~iNc)
    wNc = ndimage.binary_dilation(cloud_bool, structure=struct3) & water_bool
    w2Nc = ndimage.binary_dilation(wNc, structure=struct3) & water_bool & (~wNc)

    # --- Mixed Logic ---
    neighbor_sum = (
        iNw.astype(int) + i2Nw.astype(int) + 
        wNi.astype(int) + w2Ni.astype(int) +
        iNc.astype(int) + i2Nc.astype(int) + 
        wNc.astype(int) + w2Nc.astype(int)
    )
    mixed = (neighbor_sum >= 2)

    return {
        'iNw': iNw, 'i2Nw': i2Nw, 'i3Nw': i3Nw,
        'wNi': wNi, 'w2Ni': w2Ni, 'w3Ni': w3Ni,
        'iNc': iNc, 'i2Nc': i2Nc, 'wNc': wNc, 'w2Nc': w2Nc,
        'Mixed': mixed
    }

def create_classification(cma, landuse, lat, lat_threshold):
    """Generates the base 0-4 classification mask."""
    # 0=Skip, 1=Water, 2=Ice, 3=Cloud, 4=Land
    
    # Latitude Mask
    lat_mask = (lat >= lat_threshold)

    # Clean Cloud Mask (Remove clouds over land)
    cloud_clean = ((cma == 1) | (cma == 2))
    cloud_clean[landuse != 16] = False 
    
    cls = np.zeros_like(cma, dtype=np.uint8)
    cls[(landuse == 16) & (cma == 0)] = 1  # Water
    cls[(cma == 3)] = 2                    # Ice
    cls[cloud_clean] = 3                   # Cloud
    cls[(landuse != 16)] = 4               # Land
    cls[~lat_mask] = 0                     # Invalid Lat (set to 0/Skip)
    
    return cls, cloud_clean