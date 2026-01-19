import numpy as np
from scipy import ndimage

def compute_neighbors(classification, cloud_bool):
    struct3 = ndimage.generate_binary_structure(2, 1)
    ice_bool = (classification == 2)
    water_bool = (classification == 1)

    dilated_water = ndimage.binary_dilation(water_bool, structure=struct3)
    iNw = dilated_water & ice_bool & (~water_bool)
    i2Nw = ndimage.binary_dilation(iNw, structure=struct3) & ice_bool & (~iNw)
    i3Nw = ndimage.binary_dilation(i2Nw, structure=struct3) & ice_bool & (~iNw) & (~i2Nw)

    dilated_ice = ndimage.binary_dilation(ice_bool, structure=struct3)
    wNi = dilated_ice & water_bool & (~ice_bool)
    w2Ni = ndimage.binary_dilation(wNi, structure=struct3) & water_bool & (~wNi)
    w3Ni = ndimage.binary_dilation(w2Ni, structure=struct3) & water_bool & (~wNi) & (~w2Ni)

    iNc = ndimage.binary_dilation(cloud_bool, structure=struct3) & ice_bool
    i2Nc = ndimage.binary_dilation(iNc, structure=struct3) & ice_bool & (~iNc)
    wNc = ndimage.binary_dilation(cloud_bool, structure=struct3) & water_bool
    w2Nc = ndimage.binary_dilation(wNc, structure=struct3) & water_bool & (~wNc)

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

def create_classification(cma, landuse, lat, lon, lat_threshold, sunz, max_sunz, use_crop, crop_bounds):
    """
    Generates the base 0-4 classification mask with explicit Cropping & Sun logic.
    """
    # 0=Skip, 1=Water, 2=Ice, 3=Cloud, 4=Land
    
    # --- 1. Validity Masks ---
    if use_crop:
        # Strict mask for the box
        valid_mask = (
            (lat >= crop_bounds['lat_min']) & 
            (lat <= crop_bounds['lat_max']) & 
            (lon >= crop_bounds['lon_min']) & 
            (lon <= crop_bounds['lon_max'])
        )
    else:
        # Standard threshold
        valid_mask = (lat >= lat_threshold)
    
    if max_sunz is not None:
        day_mask = (sunz <= max_sunz)
        valid_mask = valid_mask & day_mask

    # --- 2. Surface & Cloud Definitions ---
    is_water_geo = (landuse == 16)
    is_land_geo  = (landuse != 16)

    # Cloud: CMA 1=Filled, 2=Contaminated
    cloud_raw = ((cma == 1) | (cma == 2))
    
    # Remove clouds over land
    cloud_clean = cloud_raw.copy()
    cloud_clean[is_land_geo] = False 

    # --- 3. Build Classification (Original Priority) ---
    cls = np.zeros_like(cma, dtype=np.uint8)

    cls[is_land_geo] = 4
    cls[is_water_geo & (cma == 0)] = 1
    cls[(cma == 3) & is_water_geo] = 2  # Ice is CMA=3 over water
    cls[cloud_clean] = 3

    # Apply Invalid Mask (Black)
    cls[~valid_mask] = 0
    
    return cls, cloud_clean