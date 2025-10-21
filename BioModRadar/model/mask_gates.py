import numpy as np

def mask_gates_below_ranges(radar, range_lo):
    rg = radar.range['data']
    mask = rg < range_lo
    mask = np.repeat([mask], radar.nrays, axis=0)
    for field in radar.fields.keys():
        tmp = radar.fields[field]['data'].copy()
        tmp = np.ma.masked_array(tmp, mask=mask)
        radar.fields[field]['data'] = tmp
    return radar

def mask_gates_above_ranges(radar, range_up):
    rg = radar.range['data']
    mask = rg > range_up
    mask = np.repeat([mask], radar.nrays, axis=0)
    for field in radar.fields.keys():
        tmp = radar.fields[field]['data'].copy()
        tmp = np.ma.masked_array(tmp, mask=mask)
        radar.fields[field]['data'] = tmp
    return radar

def mask_gates_outside_ranges(radar, range_lo, range_up):
    rg = radar.range['data']
    mask = np.logical_or(rg < range_lo, rg > range_up)
    mask = np.repeat([mask], radar.nrays, axis=0)
    for field in radar.fields.keys():
        tmp = radar.fields[field]['data'].copy()
        tmp = np.ma.masked_array(tmp, mask=mask)
        radar.fields[field]['data'] = tmp
    return radar

def mask_non_biological(radar, fields=None):
    if fields is None:
        radar_fields = list(radar.fields)
    else:
        if type(fields) is list:
            radar_fields = fields
        else:
            raise TypeError('"fields" must be a list.')

    radar_fields = [f for f in radar_fields if f not in ['DR', 'DR_CLASS']]
    dr_class = radar.fields['DR_CLASS']['data'].copy()
    dr_class = dr_class.filled(fill_value=0)
    mask_data = dr_class == 0

    for field in radar_fields:
        field_data = radar.fields[field]['data'].copy()
        mask_field = np.logical_or(mask_data, field_data.mask)
        # masked_data = np.ma.masked_array(field_data, mask=mask_data)
        masked_data = np.ma.masked_where(mask_data, field_data)
        radar.fields[field]['data'] = masked_data

    return radar
