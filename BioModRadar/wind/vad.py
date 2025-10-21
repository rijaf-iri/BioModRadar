import numpy as np
import pyart

def compute_vad(radar, vel_field, zlevels=None):
    if zlevels is None:
        zlevels = np.arange(50, 5000, 50)
    u_allsweeps = []
    v_allsweeps = []

    for idx in range(radar.nsweeps):
        radar_sweep = radar.extract_sweeps([idx])
        vad = pyart.retrieve.vad_browning(radar_sweep, vel_field, z_want=zlevels)

        u_allsweeps.append(vad.u_wind)
        v_allsweeps.append(vad.v_wind)

    # Average U and V over all sweeps and compute magnitude and angle
    u_avg = np.nanmean(np.array(u_allsweeps), axis=0)
    v_avg = np.nanmean(np.array(v_allsweeps), axis=0)
    direction = np.rad2deg(np.arctan2(-u_avg, -v_avg)) % 360
    speed = np.sqrt(u_avg**2 + v_avg**2)
    return zlevels, speed, direction
