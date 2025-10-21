import numpy as np
from datetime import datetime, timedelta
from netCDF4 import Dataset
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator

def make_constraint_from_era5(Grid, file_name=None, method='nearest'):
    grid_time = datetime.strptime(
        Grid['time'].attrs['units'], 'seconds since %Y-%m-%dT%H:%M:%SZ'
    )
    hour_rounded_to_nearest_1 = int(round(float(grid_time.hour)))

    if hour_rounded_to_nearest_1 == 24:
        grid_time = grid_time + timedelta(days=1)
        grid_time = datetime(
            grid_time.year,
            grid_time.month,
            grid_time.day,
            0,
            grid_time.minute,
            grid_time.second,
        )
    else:
        grid_time = datetime(
            grid_time.year,
            grid_time.month,
            grid_time.day,
            hour_rounded_to_nearest_1,
            grid_time.minute,
            grid_time.second,
        )

    ERA_grid = Dataset(file_name, mode='r')
    base_time = datetime.strptime(ERA_grid.variables['valid_time'].units,
                                 'seconds since %Y-%m-%d')

    time_step = np.argmin(np.abs(base_time - grid_time))
    # get height Geopotential/gravitational acceleration
    height_ERA = ERA_grid.variables['z'][:] / 9.80665
    u_ERA = ERA_grid.variables['u'][:]
    v_ERA = ERA_grid.variables['v'][:]
    w_ERA = ERA_grid.variables['w'][:]
    t_ERA = ERA_grid.variables['t'][:]
    ## relative humidity
    # r_ERA = ERA_grid.variables['r'][:]
    pres_ERA = ERA_grid.variables['pressure_level'][:]
    p_ERA = np.zeros(w_ERA.shape)
    for p in range(pres_ERA.shape[0]):
        p_ERA[:, p, :, :] = pres_ERA[p] * 100

    # convert Pa/s to m/s
    # Air Density rho
    rho = p_ERA / (287.058 * t_ERA)
    w_ERA = -w_ERA / (rho * 9.80665)

    lon_ERA = ERA_grid.variables['longitude'][:]
    lat_ERA = ERA_grid.variables['latitude'][:]
    u_flattened = u_ERA[time_step].flatten()
    v_flattened = v_ERA[time_step].flatten()
    w_flattened = w_ERA[time_step].flatten()

    radar_grid_lat = Grid['point_latitude'].values
    radar_grid_lon = Grid['point_longitude'].values
    radar_grid_alt = Grid['point_z'].values

    the_shape = u_ERA.shape
    lon_mgrid, lat_mgrid = np.meshgrid(lon_ERA, lat_ERA)

    lon_mgrid = np.tile(lon_mgrid, (the_shape[1], 1, 1))
    lat_mgrid = np.tile(lat_mgrid, (the_shape[1], 1, 1))
    lon_flattened = lon_mgrid.flatten()
    lat_flattened = lat_mgrid.flatten()
    height_flattened = height_ERA[time_step].flatten()
    height_flattened -= Grid.radar_altitude.values

    if method == 'nearest':
        u_interp = NearestNDInterpolator(
            (height_flattened, lat_flattened, lon_flattened), u_flattened, rescale=True
        )
        v_interp = NearestNDInterpolator(
            (height_flattened, lat_flattened, lon_flattened), v_flattened, rescale=True
        )
        w_interp = NearestNDInterpolator(
            (height_flattened, lat_flattened, lon_flattened), w_flattened, rescale=True
        )
    elif method == 'linear':
        u_interp = LinearNDInterpolator(
            (height_flattened, lat_flattened, lon_flattened), u_flattened, rescale=True
        )
        v_interp = LinearNDInterpolator(
            (height_flattened, lat_flattened, lon_flattened), v_flattened, rescale=True
        )
        w_interp = LinearNDInterpolator(
            (height_flattened, lat_flattened, lon_flattened), w_flattened, rescale=True
        )
    else:
        raise NotImplementedError('%s interpolation method not implemented!' % method)

    u_new = u_interp(radar_grid_alt, radar_grid_lat, radar_grid_lon)
    v_new = v_interp(radar_grid_alt, radar_grid_lat, radar_grid_lon)
    w_new = w_interp(radar_grid_alt, radar_grid_lat, radar_grid_lon)

    # Free up memory
    ERA_grid.close()

    u_field = {}
    u_field['standard_name'] = 'u_wind'
    u_field['long_name'] = 'meridional component of wind velocity'
    v_field = {}
    v_field['standard_name'] = 'v_wind'
    v_field['long_name'] = 'zonal component of wind velocity'
    w_field = {}
    w_field['standard_name'] = 'w_wind'
    w_field['long_name'] = 'vertical component of wind velocity'
    Grid['u'] = xr.DataArray(
        np.expand_dims(u_new, 0), dims=('time', 'z', 'y', 'x'), attrs=u_field
    )
    Grid['v'] = xr.DataArray(
        np.expand_dims(v_new, 0), dims=('time', 'z', 'y', 'x'), attrs=v_field
    )
    Grid['w'] = xr.DataArray(
        np.expand_dims(w_new, 0), dims=('time', 'z', 'y', 'x'), attrs=w_field
    )
    return Grid

