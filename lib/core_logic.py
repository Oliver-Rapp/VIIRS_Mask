import numpy as np
from scipy import ndimage
from . import config

def compute_neighbors(classification, cloud_bool):
    struct3 = ndimage.generate_binary_structure(2, 1)
    
    ice_bool = (classification == 2)
    water_bool = (classification == 1)

    # ==========================================
    # 1. ICE INTERACTIONS
    # ==========================================
    
    # --- Ice near Water (InW) ---
    d_water = ndimage.binary_dilation(water_bool, structure=struct3)
    iNw = d_water & ice_bool & (~water_bool)
    
    d_iNw = ndimage.binary_dilation(iNw, structure=struct3)
    i2Nw = d_iNw & ice_bool & (~iNw)
    
    d_i2Nw = ndimage.binary_dilation(i2Nw, structure=struct3)
    i3Nw = d_i2Nw & ice_bool & (~iNw) & (~i2Nw)

    # --- Ice near Cloud (InC) ---
    d_cloud = ndimage.binary_dilation(cloud_bool, structure=struct3)
    iNc = d_cloud & ice_bool
    
    d_iNc = ndimage.binary_dilation(iNc, structure=struct3)
    i2Nc = d_iNc & ice_bool & (~iNc)
    
    d_i2Nc = ndimage.binary_dilation(i2Nc, structure=struct3)
    i3Nc = d_i2Nc & ice_bool & (~iNc) & (~i2Nc)

    # ==========================================
    # 2. WATER INTERACTIONS
    # ==========================================

    # --- Water near Ice (WnI) ---
    d_ice = ndimage.binary_dilation(ice_bool, structure=struct3)
    wNi = d_ice & water_bool & (~ice_bool)
    
    d_wNi = ndimage.binary_dilation(wNi, structure=struct3)
    w2Ni = d_wNi & water_bool & (~wNi)
    
    d_w2Ni = ndimage.binary_dilation(w2Ni, structure=struct3)
    w3Ni = d_w2Ni & water_bool & (~wNi) & (~w2Ni)

    # --- Water near Cloud (WnC) ---
    # d_cloud already computed
    wNc = d_cloud & water_bool
    
    d_wNc = ndimage.binary_dilation(wNc, structure=struct3)
    w2Nc = d_wNc & water_bool & (~wNc)
    
    d_w2Nc = ndimage.binary_dilation(w2Nc, structure=struct3)
    w3Nc = d_w2Nc & water_bool & (~wNc) & (~w2Nc)

    # ==========================================
    # 3. CLOUD INTERACTIONS
    # ==========================================
    
    # --- Cloud near Ice (CnI) ---
    # d_ice already computed
    cNi = d_ice & cloud_bool
    
    d_cNi = ndimage.binary_dilation(cNi, structure=struct3)
    c2Ni = d_cNi & cloud_bool & (~cNi)
    
    d_c2Ni = ndimage.binary_dilation(c2Ni, structure=struct3)
    c3Ni = d_c2Ni & cloud_bool & (~cNi) & (~c2Ni)

    # --- Cloud near Water (CnW) ---
    # d_water already computed
    cNw = d_water & cloud_bool
    
    d_cNw = ndimage.binary_dilation(cNw, structure=struct3)
    c2Nw = d_cNw & cloud_bool & (~cNw)
    
    d_c2Nw = ndimage.binary_dilation(c2Nw, structure=struct3)
    c3Nw = d_c2Nw & cloud_bool & (~cNw) & (~c2Nw)

    # ==========================================
    # 4. AGGREGATE & EXPORT
    # ==========================================

    # Keys updated to requested notation
    masks = {
        'InW': iNw, 'I2nW': i2Nw, 'I3nW': i3Nw,
        'InC': iNc, 'I2nC': i2Nc, 'I3nC': i3Nc,
        
        'WnI': wNi, 'W2nI': w2Ni, 'W3nI': w3Ni,
        'WnC': wNc, 'W2nC': w2Nc, 'W3nC': w3Nc,
        
        'CnI': cNi, 'C2nI': c2Ni, 'C3nI': c3Ni,
        'CnW': cNw, 'C2nW': c2Nw, 'C3nW': c3Nw
    }

    # --- Mixed Logic ---
    neighbor_sum = np.zeros_like(classification, dtype=int)
    
    # Sum up all masks
    for k in masks:
        neighbor_sum += masks[k].astype(int)
        
    masks['Mixed'] = (neighbor_sum >= 2)

    # --- Interior (Pure) Logic ---
    # Pure pixels must not be any edge type (sum == 0)
    masks['IN'] = ice_bool & (neighbor_sum == 0)
    masks['WN'] = water_bool & (neighbor_sum == 0)
    masks['CN'] = cloud_bool & (neighbor_sum == 0)

    return masks

def create_classification(cma, landuse, lat, sunz):
    day_mask = (sunz <= config.MAX_SOLAR_ZENITH)
    
    is_water_geo = (landuse == 16)
    is_land_geo  = (landuse != 16)

    # 1 (Cloudy) and 2 (Contaminated) used for clouds
    cloud_raw = ((cma == 1) | (cma == 2))
    cloud_clean = cloud_raw.copy()
    cloud_clean[is_land_geo] = False 

    cls = np.zeros_like(cma, dtype=np.uint8)
    cls[is_water_geo] = 1
    cls[is_land_geo] = 4
    cls[is_water_geo & (cma == 3)] = 2
    cls[cloud_clean] = 3
    
    cls[~day_mask] = 0
    cls[cma == 255] = 0
    cls[is_land_geo] = 4 
    
    return cls, cloud_clean