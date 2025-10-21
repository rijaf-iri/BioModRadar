import os
import re
import tempfile
import shutil
import requests
import numpy as np
import xarray as xr
import getgfs
# import cfgrib
from datetime import datetime, timedelta, timezone
from netCDF4 import Dataset, num2date
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from .util import *

def make_constraint_from_gfs(Grid, method='nearest'):
    grid_time = cftime2datetime(Grid.time.values[0])

    bbox = {}
    bbox['north'] = round(Grid['point_latitude'].values.max(), 2)
    bbox['south'] = round(Grid['point_latitude'].values.min(), 2)
    bbox['east'] = round(Grid['point_longitude'].values.max(), 2)
    bbox['west'] = round(Grid['point_longitude'].values.min(), 2)

    wind_data = download_gfs_forecast(grid_time, bbox)

    wind_vars = {}
    wind_vars['u'] = {
            'standard_name': 'u_wind',
            'long_name': 'meridional component of wind velocity'
            }
    wind_vars['v'] = {
            'standard_name': 'v_wind',
            'long_name': 'zonal component of wind velocity'
            }
    wind_vars['w'] = {
            'standard_name': 'w_wind',
            'long_name': 'vertical component of wind velocity'
            }

    radar_grid_lat = Grid['point_latitude'].values
    radar_grid_lon = Grid['point_longitude'].values
    radar_grid_alt = Grid['point_z'].values

    for k in wind_vars:
        p_v = wind_data[k]['pres']
        p_z = wind_data['z']['pres']
        p, i, j = np.intersect1d(p_z, p_v, return_indices=True)
        z = wind_data['z']['data'][:, i, :, :]

        v_shape = wind_data[k]['data'].shape
        mlon, mlat = np.meshgrid(wind_data[k]['lon'], wind_data[k]['lat'])
        mlon = np.tile(mlon, (v_shape[1], 1, 1))
        mlat = np.tile(mlat, (v_shape[1], 1, 1))
        flon = mlon.flatten()
        flat = mlat.flatten()
        fhght = z.flatten()
        fhght -= Grid.radar_altitude.values
        var = wind_data[k]['data'].flatten()

        if method == 'nearest':
            f_interp = NearestNDInterpolator((fhght, flat, flon), var, rescale=True)
        elif method == 'linear':
            f_interp = LinearNDInterpolator((fhght, flat, flon), u_flattened, rescale=True)
        else:
            raise NotImplementedError('%s interpolation method not implemented!' % method)

        v_new = f_interp(radar_grid_alt, radar_grid_lat, radar_grid_lon)
        Grid[f'{k.upper()}_gfs'] = xr.DataArray(np.expand_dims(v_new, 0),
                                                dims=('time', 'z', 'y', 'x'),
                                                attrs=wind_vars[k])
    return Grid

def download_gfs_forecast(date_time, bbox):
    if not isinstance(date_time, datetime):
        date_time = datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S')
    # date_now = datetime.now(timezone.utc)
    date_now = datetime.utcnow()
    date_diff = date_now - date_time
    if date_diff.days > 10:
        wind_data = download_gfs_forecast_archive(date_time, bbox)
    else:
        wind_data = download_gfs_forecast_getgfs(date_time, bbox)
        # wind_data = download_gfs_forecast_production(date_time, bbox)

    return wind_data

def download_gfs_forecast_archive(date_time, bbox):
    base_url = 'https://tds.gdex.ucar.edu/thredds/ncss/grid/files/g/d084001'
    frmt_url = '{year}/{date}/gfs.0p25.{date}{run}.f{hour}.grib2'
    down_vars = {'u': 'u-component_of_wind_isobaric',
                 'v': 'v-component_of_wind_isobaric',
                 'w': 'Vertical_velocity_geometric_isobaric',
                 'z': 'Geopotential_height_isobaric'}

    date_info = _gfs_forecast_datetime(date_time, 3)
    date_url = frmt_url.format(
        year = date_info['year'],
        date = date_info['date'],
        run = date_info['run'],
        hour = date_info['hour'])
    params = round_bounding_box(bbox, 0.25).copy()
    params['var'] = [down_vars[k] for k in down_vars]
    params['time'] = date_info['time']
    params.update({'horizStride': 1,
            'vertStride': 1,
            'accept': 'netcdf4-classic',
            'addLatLon': 'true'})
    res = requests.get(f'{base_url}/{date_url}', params=params)
    if res.status_code != 200:
        raise Exception(f'Unable to download {date_url}')

    filename = res.headers.get('Content-Disposition')
    filename = filename.split('filename=')[-1]
    tmp_dir = tempfile.mkdtemp()
    ncfile = os.path.join(tmp_dir, filename)
    with open(ncfile, 'wb') as file:
        file.write(res.content)

    nc_data = Dataset(ncfile, mode='r')
    out = {k:_get_gfs_data_archive(nc_data, v) for k, v in down_vars.items()}
    nc_data.close()
    os.remove(ncfile)
    shutil.rmtree(tmp_dir)
    return out

def download_gfs_forecast_getgfs(date_time, bbox):
    if not isinstance(date_time, datetime):
       date_time = datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S')

    date_time = date_time.replace(minute=0, second=0, microsecond=0)
    date_time = date_time.strftime('%Y-%m-%d %H:%M:%S')
    down_vars = {'u': 'ugrdprs',
                 'v': 'vgrdprs',
                 'w': 'dzdtprs',
                 'z': 'hgtprs'}
    fcst = getgfs.Forecast('0p25', '1hr')
    bbox = round_bounding_box(bbox, 0.25).copy()
    blat = f'[{bbox['south']}:{bbox['north']}]'
    blon = f'[{bbox['west']}:{bbox['east']}]'
    params = [down_vars[k] for k in down_vars]
    res = fcst.get(params, date_time, lat=blat, lon=blon)
    time = fcst.datetime_to_forecast(date_time)
    out = {k:_get_gfs_data_getgfs(res, v, time) for k, v in down_vars.items()}
    return out

def download_gfs_forecast_production(date_time, bbox):
    base_url = 'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl'
    dir_frmt='/gfs.{date}/{run}/atmos'
    file_frmt='gfs.t{run}z.pgrb2.0p25.f{hour}'

    date_info = _gfs_forecast_datetime(date_time, 1)
    dir_url = dir_frmt.format(date=date_info['date'], run=date_info['run'])
    file_url = file_frmt.format(run=date_info['run'], hour=date_info['hour'])
    params = {'dir': dir_url, 'file': file_url}

    check = _check_gfs_forecast_avail(base_url, params)
    if check == 'net':
        raise Exception(f'Unable to download {dir_url}/{file_url}')

    if check == 'not':
        new_time = date_time - timedelta(hours=6)
        date_info = _gfs_forecast_datetime(date_time, 1, new_time.hour)
        dir_url = dir_frmt.format(date=date_info['date'], run=date_info['run'])
        file_url = file_frmt.format(run=date_info['run'], hour=date_info['hour'])
        params = {'dir': dir_url, 'file': file_url}

    down_vars = {
            'u': 'var_UGRD',
            'v': 'var_VGRD',
            'w': 'var_VVEL',
            'z': 'var_HGT',
            't': 'var_TMP'
            }
    params.update({down_vars[k]: 'on' for k in down_vars})

    pres_lev = [1000, 975, 950, 925, 900, 850, 800, 750,
                700, 650, 600, 550, 500, 450, 400, 350,
                300, 250, 200, 150, 100, 70, 50, 40, 30,
                20, 15, 10, 7, 5, 3, 2, 1,
                0.7, 0.4, 0.2, 0.1, 0.07, 0.04, 0.02, 0.01]
    params.update({f'lev_{p}_mb': 'on' for p in pres_lev})

    rbbox = round_bounding_box(bbox.copy(), 0.25)
    params.update({
        'subregion': '',
        'leftlon': rbbox['west'],
        'rightlon': rbbox['east'],
        'toplat': rbbox['north'],
        'bottomlat': rbbox['south'],
        })

    res = requests.get(base_url, params=params)
    if res.status_code != 200:
        raise Exception(f'Unable to download {dir_url}/{file_url}')

    filename = res.headers.get('Content-Disposition')
    filename = filename.split('filename=')[-1].strip('"')
    tmp_dir = tempfile.mkdtemp()
    gribfile = os.path.join(tmp_dir, filename)
    with open(gribfile, 'wb') as file:
        file.write(res.content)

    xr_data = xr.open_dataset(gribfile, engine='cfgrib',
                              filter_by_keys={'typeOfLevel': 'isobaricInPa'})
    ## check if w and t have same pressure level
    # pres_t = xr_data[xr_data['t'].dims[0]].values
    # pres_w = xr_data[xr_data['w'].dims[0]].values
    # if not np.array_equal(pres_w, pres_t):
    pres_w = xr_data['isobaricInhPa'].values
    ## convert Pa/s to m/s
    pres = np.zeros(xr_data['w'].shape)
    for p in range(pres_w.shape[0]):
        pres[p, :, :] = pres_w[p] * 100
    rho = pres / (287.058 * xr_data['t'].values)
    xr_data['w'].data = -xr_data['w'].values / (rho * 9.80665)
    
    pvars = {'u': 'u', 'v': 'v', 'w': 'w', 'z': 'gh'}
    out = {k:_get_gfs_data_production(xr_data, v) for k, v in pvars.items()}
    os.remove(gribfile)
    shutil.rmtree(tmp_dir)
    return out

def _gfs_forecast_datetime(date_time, tstep=1, fcst_run=None):
    hour = date_time.hour
    hours_run = np.array([0, 6, 12, 18, 24])
    hours_fcst = np.arange(0, 48, tstep)
    if fcst_run is None:
        irun = np.searchsorted(hours_run, hour, side='right')
        fcst_run = hours_run[irun - 1]
    ifst = np.searchsorted(hours_fcst, hour - fcst_run, side='right')
    fcst_hour = hours_fcst[ifst - 1]
    this_date = date_time.replace(hour=int(fcst_run), minute=0, second=0)
    this_time = this_date + timedelta(hours=int(fcst_hour))
    return {
        'year': date_time.year,
        'date': date_time.strftime('%Y%m%d'),
        'run': f'{fcst_run:02}',
        'hour': f'{fcst_hour:03}',
        'time': this_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    }

def _check_gfs_forecast_avail(base_url, params):
    params0 = params.copy()
    params0.update({
        'var_UGRD': 'on',
        'lev_1000_mb': 'on',
        'subregion': '',
        'toplat': 0,
        'leftlon': 30,
        'rightlon': 30.25,
        'bottomlat': 0.25
    })
    check = requests.get(base_url, params=params0)
    if check.status_code == 200:
        ret = 'ok'
    elif check.status_code == 404:
        ret = 'not'
    else:
        ret = 'net'
    return ret

def _get_gfs_data_archive(nc_data, var):
    crd = nc_data.variables[var].coordinates.strip()
    crd = [s for s in crd.split(' ') if s != 'reftime']
    time = num2date(times=nc_data[crd[0]][:], units=nc_data[crd[0]].units)
    time = cftime2datetime(time[0])
    pres = nc_data[crd[1]][:].filled()/100
    lat = nc_data[crd[2]][:].filled()
    lon = nc_data[crd[3]][:].filled()
    data = nc_data[var][:].filled(np.nan)
    return {'time': time, 'pres': pres,
            'lon': lon, 'lat':lat,
            'data': data}

def _get_gfs_data_getgfs(res, var, fcst_time):
    # time = res.variables[var].coords['time'].values[0]
    # time = num2date(times=time, units='days since 1-1-1 00:00:0.0')
    # time = cftime2datetime(time)
    # time = round_to_nearest_hour(time)
    time = datetime.strptime(f'{fcst_time[0]}{fcst_time[1]}', '%Y%m%d%H')
    tt = re.findall(r'\[(.*?)\]', fcst_time[2])
    time = time + timedelta(hours=int(tt[0]))
    pres = np.array(res.variables[var].coords['lev'].values)
    lat = np.array(res.variables[var].coords['lat'].values)
    lon = np.array(res.variables[var].coords['lon'].values)
    data = res.variables[var].data
    return {'time': time, 'pres': pres,
            'lon': lon, 'lat':lat,
            'data': data}

def _get_gfs_data_production(xr_data, var):
    crd = xr_data[var].dims
    time = xr_data['valid_time'].values
    # time = xr_data['time'].values + xr_data['step'].values
    time = npdt64todatetime(time)
    pres = xr_data[crd[0]].values
    lat = xr_data[crd[1]].values
    lon = xr_data[crd[2]].values
    data = xr_data[var].values
    data = np.expand_dims(data, axis=0)
    return {'time': time, 'pres': pres,
            'lon': lon, 'lat':lat,
            'data': data}
