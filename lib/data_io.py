import os
import re
import numpy as np
from netCDF4 import Dataset
from . import config

def get_file_groups(directory):
    groups = {}
    files = [f for f in os.listdir(directory) if f.endswith('.nc')]
    
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
            
    return groups

def load_scene_data(files):
    if not all(k in files for k in ['cma', 'geo', 'l1b', 'tex']): return None

    try:
        ds_l1b = Dataset(files['l1b'], 'r')
        
        # --- 1. Load Navigation ---
        lat = None
        for v in ['lat', 'latitude', 'nav_lat']:
            if v in ds_l1b.variables:
                lat = np.squeeze(ds_l1b.variables[v][:])
                break
        if lat is None: return None

        lon_var = next((k for k in ['lon', 'longitude', 'nav_lon'] if k in ds_l1b.variables), None)
        lon = np.squeeze(ds_l1b.variables[lon_var][:]) if lon_var else np.zeros_like(lat)

        # --- 2. Calculate Crop Slice ---
        # Initialize slice to full image
        ys = slice(None); xs = slice(None)
        
        if config.USE_CROP:
            bounds = config.CROP_BOUNDS
            
            # Create a boolean mask for the valid region
            region_mask = (
                (lat >= bounds['lat_min']) & 
                (lat <= bounds['lat_max']) & 
                (lon >= bounds['lon_min']) & 
                (lon <= bounds['lon_max'])
            )
            
            # If no part of the scene is in the box, skip it
            if not np.any(region_mask):
                return None
            
            # Find the bounding box of the valid region to slice efficiently
            rows, cols = np.where(region_mask)
            y_min, y_max = rows.min(), rows.max() + 1
            x_min, x_max = cols.min(), cols.max() + 1
            
            # Update slices
            ys = slice(y_min, y_max)
            xs = slice(x_min, x_max)
            
            # Apply slice to nav data immediately
            lat = lat[ys, xs]
            lon = lon[ys, xs]
        else:
            # If not cropping, apply simple threshold check
            if np.max(lat) < config.LATITUDE_THRESHOLD:
                return None

        # --- 3. Load & Slice Variables ---
        t11 = ds_l1b.variables['image3'][0, ys, xs]
        t12 = ds_l1b.variables['image4'][0, ys, xs]
        t37 = ds_l1b.variables['image5'][0, ys, xs]
        t87 = ds_l1b.variables['image7'][0, ys, xs]
        sunz = ds_l1b.variables['sunzenith'][0, ys, xs]
        satz = ds_l1b.variables['satzenith'][0, ys, xs]

        ds_cma = Dataset(files['cma'], 'r')
        cma = ds_cma.variables['cma_extended'][0, ys, xs]

        ds_geo = Dataset(files['geo'], 'r')
        landuse = ds_geo.variables['landuse'][0, ys, xs]

        ds_tex = Dataset(files['tex'], 'r')
        r06_tex = ds_tex.variables['r06'][0, ys, xs].astype(float)
        t11_tex = ds_tex.variables['t11'][0, ys, xs].astype(float)
        t11t12_tex = ds_tex.variables['t11t12'][0, ys, xs].astype(float)
        t37t12_tex = ds_tex.variables['t37t12'][0, ys, xs].astype(float)

        data = {
            'lat': lat, 'lon': lon,
            'cma': cma, 'landuse': landuse,
            't11': t11, 'sunz': sunz, 'satz': satz,
            'diff1': t11 - t12, 'diff2': t37 - t12, 'diff3': t87 - t12,
            'r06_tex': r06_tex, 't11_tex': t11_tex,
            't11t12_tex': t11t12_tex, 't37t12_tex': t37t12_tex
        }
        
        ds_l1b.close(); ds_cma.close(); ds_geo.close(); ds_tex.close()
        return data

    except Exception:
        return None