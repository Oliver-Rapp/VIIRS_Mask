import os
import re
import numpy as np
from netCDF4 import Dataset
from . import config

def get_file_groups(directory):
    """Scans directory and groups files by timestamp using standard Regex."""
    groups = {}
    files = [f for f in os.listdir(directory) if f.endswith('.nc')]
    
    for f in files:
        # Regex to match ISO 8601 timestamps in filename
        match = re.search(r'_(\d{8}T\d{7}Z)_', f) 
        if not match: continue
        
        ts = match.group(1)
        if ts not in groups: groups[ts] = {}
        
        full_path = os.path.join(directory, f)
        if 'S_NWC_CMA_' in f: groups[ts]['cma'] = full_path
        elif 'physiography' in f: groups[ts]['geo'] = full_path
        elif 'viirs' in f: groups[ts]['l1b'] = full_path
        elif 'textures' in f: groups[ts]['tex'] = full_path
            
    return groups

def load_scene_data(files):
    """Opens files, checks lat threshold, and returns dictionary of arrays."""
    # Check completeness
    if not all(k in files for k in ['cma', 'geo', 'l1b', 'tex']): return None

    ds_l1b = None; ds_cma = None; ds_geo = None; ds_tex = None

    try:
        ds_l1b = Dataset(files['l1b'], 'r')
        
        # --- 1. Navigation Check ---
        lat = None
        for v in ['lat', 'latitude', 'nav_lat']:
            if v in ds_l1b.variables:
                lat = np.squeeze(ds_l1b.variables[v][:])
                break
        
        if lat is None: return None
        
        # Immediate skip if the whole scene is too far south
        if np.max(lat) < config.LATITUDE_THRESHOLD: 
            return None

        lon_var = next((k for k in ['lon', 'longitude', 'nav_lon'] if k in ds_l1b.variables), None)
        if lon_var is None:
            # Create dummy lon if missing, though typically needed for Mosaic
            lon = np.zeros_like(lat) 
        else:
            lon = np.squeeze(ds_l1b.variables[lon_var][:])

        # --- 2. Load Variables ---
        t11 = ds_l1b.variables['image3'][0, :, :]
        t12 = ds_l1b.variables['image4'][0, :, :]
        t37 = ds_l1b.variables['image5'][0, :, :]
        t87 = ds_l1b.variables['image7'][0, :, :]
        sunz = ds_l1b.variables['sunzenith'][0, :, :]
        satz = ds_l1b.variables['satzenith'][0, :, :]

        ds_cma = Dataset(files['cma'], 'r')
        cma = ds_cma.variables['cma_extended'][0, :, :]

        ds_geo = Dataset(files['geo'], 'r')
        landuse = ds_geo.variables['landuse'][0, :, :]

        ds_tex = Dataset(files['tex'], 'r')
        r06_tex = ds_tex.variables['r06'][0,:,:].astype(float)
        t11_tex = ds_tex.variables['t11'][0,:,:].astype(float)
        t11t12_tex = ds_tex.variables['t11t12'][0,:,:].astype(float)
        t37t12_tex = ds_tex.variables['t37t12'][0,:,:].astype(float)

        # Pack into dictionary
        data = {
            'lat': lat, 'lon': lon,
            'cma': cma, 'landuse': landuse,
            't11': t11, 'sunz': sunz, 'satz': satz,
            'diff1': t11 - t12,
            'diff2': t37 - t12,
            'diff3': t87 - t12,
            'r06_tex': r06_tex, 't11_tex': t11_tex,
            't11t12_tex': t11t12_tex, 't37t12_tex': t37t12_tex
        }
        return data

    except Exception:
        # Fail silently on bad files to keep the pool running
        return None
    finally:
        # Cleanup
        for d in [ds_l1b, ds_cma, ds_geo, ds_tex]:
            try: 
                if d: d.close()
            except: pass