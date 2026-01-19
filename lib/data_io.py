import os
import re
import numpy as np
from netCDF4 import Dataset
from . import config

def get_file_groups(directory):
    """
    Scans directory and groups files by their timestamp.
    Expects files like: S_NWC_CMA_..._20250415T105803Z_...nc
    """
    groups = {}
    if not os.path.exists(directory):
        print(f"ERROR: Directory not found: {directory}")
        return {}

    files = sorted([f for f in os.listdir(directory) if f.endswith('.nc')])
    
    for f in files:
        match = re.search(r'_(\d{8}T\d{7}Z)_', f) 
        if not match: continue
        
        ts = match.group(1)
        if ts not in groups: groups[ts] = {}
        path = os.path.join(directory, f)
        
        if 'S_NWC_CMA_' in f: groups[ts]['cma'] = path
        elif 'physiography' in f: groups[ts]['geo'] = path
        elif 'viirs' in f: groups[ts]['l1b'] = path
        elif 'textures' in f: groups[ts]['tex'] = path
            
    # Return only complete groups
    valid_groups = {k: v for k, v in groups.items() if len(v) == 4}
    return valid_groups

def load_scene_data(files):
    """
    Loads data using bounding box slicing to save memory.
    """
    try:
        # 1. Open Navigation to determine Crop Slice
        with Dataset(files['l1b'], 'r') as ds_l1b:
            lat_var = next((v for v in ['lat', 'latitude', 'nav_lat'] if v in ds_l1b.variables), None)
            lon_var = next((v for v in ['lon', 'longitude', 'nav_lon'] if v in ds_l1b.variables), None)
            
            if not lat_var: return None

            lat_full = np.squeeze(ds_l1b.variables[lat_var][:])
            lon_full = np.squeeze(ds_l1b.variables[lon_var][:]) if lon_var else np.zeros_like(lat_full)
            
            # --- CALCULATE SLICE ---
            if config.USE_CROP:
                mask = (
                    (lat_full >= config.CROP_BOUNDS['lat_min']) &
                    (lat_full <= config.CROP_BOUNDS['lat_max']) &
                    (lon_full >= config.CROP_BOUNDS['lon_min']) &
                    (lon_full <= config.CROP_BOUNDS['lon_max'])
                )
                
                if not np.any(mask): 
                    return None # Scene not in Fram Strait

                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                
                ys = slice(rmin, rmax + 1)
                xs = slice(cmin, cmax + 1)
                
                lat = lat_full[ys, xs]
                lon = lon_full[ys, xs]
            else:
                if np.max(lat_full) < config.LATITUDE_THRESHOLD: return None
                ys = slice(None); xs = slice(None)
                lat = lat_full; lon = lon_full

            # Load Variables (masked=True handles fill values automatically)
            t11 = ds_l1b.variables['image3'][0, ys, xs]
            t12 = ds_l1b.variables['image4'][0, ys, xs]
            t37 = ds_l1b.variables['image5'][0, ys, xs]
            t87 = ds_l1b.variables['image7'][0, ys, xs]
            sunz = ds_l1b.variables['sunzenith'][0, ys, xs]
            satz = ds_l1b.variables['satzenith'][0, ys, xs]

        with Dataset(files['cma'], 'r') as ds:
            cma = ds.variables['cma_extended'][0, ys, xs]

        with Dataset(files['geo'], 'r') as ds:
            landuse = ds.variables['landuse'][0, ys, xs]

        with Dataset(files['tex'], 'r') as ds:
            # Load textures as float
            r06_tex = ds.variables['r06'][0, ys, xs].astype(float)
            t11_tex = ds.variables['t11'][0, ys, xs].astype(float)
            t11t12_tex = ds.variables['t11t12'][0, ys, xs].astype(float)
            t37t12_tex = ds.variables['t37t12'][0, ys, xs].astype(float)

        return {
            'lat': lat, 'lon': lon,
            'cma': cma, 'landuse': landuse,
            't11': t11, 'sunz': sunz, 'satz': satz,
            'diff1': t11 - t12,
            'diff2': t37 - t12,
            'diff3': t87 - t12,
            'r06_tex': r06_tex, 't11_tex': t11_tex,
            't11t12_tex': t11t12_tex, 't37t12_tex': t37t12_tex
        }

    except Exception as e:
        print(f"Data Load Error: {e}")
        return None