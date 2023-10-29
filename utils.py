import numpy as np
import os
import json
from datetime import datetime
import cv2

from PIL import Image
import config as cfg


def read_image_and_meta(path):
    
    im = Image.open(path)
    img = np.asarray(im)
    metastr = im.info['metadata'].replace('\'', "\"")
    try:
        metadata = json.loads(metastr)
        
        return img, metadata['PixHawk_data']
    except ValueError:
        print('JSONDecodeError')
        
        return img, None

    
def from_meta_back(meta, packettype, datatypes):
    
    data = []
    for d in meta:
        if packettype == d['mavpackettype']:
            for t in datatypes:
                data.append(d[t]) 
            break

    return np.asarray(data)


def from_meta(meta, packettype, datatypes):
    
    data = []
    if type(meta)==list:
        for d in meta:
            if packettype == d['mavpackettype']:
                for t in datatypes:
                    data.append(d[t]) 
                break
    elif type(meta)==dict:
        for t in datatypes:
            data.append(meta[packettype][t])

    return np.asarray(data)


def pt2h(abs_pressure, P0, temperature):
    
    return (1 - abs_pressure/P0) * 8.3144598 * (273.15 + temperature/100) / 9.80665 / 0.0289644


def read_pos_data(path : str, P0 : float) -> dict:
    """
    args:
        path to png with meta
        P0 - pressure at drone start
    
    """
    frame, meta = read_image_and_meta(path)
    
    timestamp = from_meta(meta, 'SYSTEM_TIME', ['time_boot_ms'])
    
    lat, lon, rel_alt, vx, vy, vz = from_meta(meta, 'GLOBAL_POSITION_INT', ['lat', 'lon', 'relative_alt', 'vx', 'vy', 'vz'])
    
    press_abs, temperature = from_meta(meta, 'SCALED_PRESSURE', ['press_abs', 'temperature'])
    altitude = pt2h(press_abs, P0, temperature)

    angles = from_meta(meta, 'ATTITUDE', ['roll', 'pitch', 'yaw'])
            
    heading = from_meta(meta, 'VFR_HUD', ['heading']) / 180 * np.pi
    
    return dict(
                timestamp=float(timestamp)/1000,
                image=frame,
                lat=lat/10**7,
                lon=lon/10**7,
                rel_alt=rel_alt,
                vels = [vx, vy, vz],
                pressure=press_abs,
                temperature=temperature,
                altitude=altitude,
                roll=float(angles[0]),
                pitch=float(angles[1]),
                yaw=float(angles[2]),
                heading=float(heading),
                dpp=cfg.TEST_DPP - angles[:2]*cfg.focal,
               )


def to_homo(arr):
    
    if len(arr.shape)==1:
        
        return np.hstack((arr, 1))
    else:
        
        homo = np.ones((len(arr),1), dtype=arr.dtype)
        
        return np.hstack((arr, homo))