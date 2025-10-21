import numpy as np

def classify_mistnet(radar):
    weather = radar.fields['WEATHER']['data'].copy()
    biology = radar.fields['BIOLOGY']['data'].copy()
    cell = radar.fields['CELL']['data'].copy()
    background = radar.fields['BACKGROUND']['data'].copy()

    mask = np.logical_or(background > 0.9,  np.isnan(background))
    mask = mask.filled()
    weather.mask = np.ma.mask_or(weather.mask, mask)
    biology.mask = np.ma.mask_or(biology.mask, mask)
    cell.mask = np.ma.mask_or(cell.mask, mask)

    mist_class = np.empty(biology.shape, dtype=int)
    mist_class = np.ma.masked_array(mist_class, mask=weather.mask)
    mask_weather = np.logical_or(np.logical_or(weather > 0.45, cell >= 1), biology < 0.5)
    mist_class[~mask_weather] = 1
    mist_class[mask_weather] = 0

    class_dict = {
        'data': mist_class,
        'units': '',
        'long_name': 'MISTNET Classification: weather=0, biology=1',
        '_FillValue': -9999,
        'standard_name': 'mistnet_class',
    }
    radar.add_field('MIST_CLASS', class_dict, replace_existing=True)
    return radar

def classify_radxpid(radar):
    pid = radar.fields["PID"]["data"].copy()
    pid = np.ma.masked_equal(pid, 0)
    pid = np.ma.masked_greater_equal(pid, 16)
    # pid = np.ma.masked_equal(pid, 16)
    # # pid = np.ma.masked_equal(pid, 17)

    pid_class = np.empty(pid.shape, dtype=int)
    pid_class = np.ma.masked_array(pid_class, mask=pid.mask)
    pid_class[pid < 15] = 0
    pid_class[pid == 15] = 1
    # pid_class[pid >= 15] = 1

    class_dict = {
        'data': pid_class,
        'units': '',
        'long_name': 'Lrose-RadxPid Classification: weather=0, biology=1',
        '_FillValue': -9999,
        'standard_name': 'pid_class',
    }
    radar.add_field('PID_CLASS', class_dict, replace_existing=True)
    return radar
