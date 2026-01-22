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

def load_var(ds, var_name, ys, xs):
    """
    Safely loads a variable from a NetCDF dataset.
    Handles scaling/offset (via netCDF4) and converts MaskedArrays to NaN.
    """
    if var_name not in ds.variables:
        # Return a block of NaNs if variable is missing
        shape = (ys.stop - ys.start, xs.stop - xs.start)
        return np.full(shape, np.nan)

    data = ds.variables[var_name][0, ys, xs]
    
    # If the data is a MaskedArray (contains _FillValue), convert fill values to NaN
    if np.ma.is_masked(data):
        return data.filled(np.nan)
    
    return data

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

            # Load Variables using safe loader
            t11 = load_var(ds_l1b, 'image3', ys, xs)
            t12 = load_var(ds_l1b, 'image4', ys, xs)
            t37 = load_var(ds_l1b, 'image5', ys, xs)
            t87 = load_var(ds_l1b, 'image7', ys, xs)
            sunz = load_var(ds_l1b, 'sunzenith', ys, xs)
            satz = load_var(ds_l1b, 'satzenith', ys, xs)

        with Dataset(files['cma'], 'r') as ds:
            # CMA is usually byte data, keep raw (masked handled by fill value 255 later)
            cma = ds.variables['cma_extended'][0, ys, xs]

        with Dataset(files['geo'], 'r') as ds:
            landuse = ds.variables['landuse'][0, ys, xs]

        with Dataset(files['tex'], 'r') as ds:
            # Load textures, convert masked to NaN automatically
            r06_tex = load_var(ds, 'r06', ys, xs)
            t11_tex = load_var(ds, 't11', ys, xs)
            t11t12_tex = load_var(ds, 't11t12', ys, xs)
            t37t12_tex = load_var(ds, 't37t12', ys, xs)

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