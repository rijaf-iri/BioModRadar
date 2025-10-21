import copy
import numpy as np
from skimage import measure
from skimage.morphology import remove_small_objects

def radar_blob_segmentation(radar, sweeps, field, min_field=None,
                            min_size=100, connectivity=2):
    radar_c = copy.deepcopy(radar)
    out_mask = []
    for s in sweeps:
        radar_s = radar_c.extract_sweeps([s])
        data = radar_s.fields[field]['data']
        data = data.filled(fill_value=np.nan)
        if min_field is None:
            mask = np.isfinite(data)
        else:
            mask = np.isfinite(data) & (data > min_field)
        mask = remove_small_objects(mask.astype(bool),
                                    min_size=min_size,
                                    connectivity=connectivity)
        out_mask += [mask]
    blob_mask = ~np.concatenate(out_mask, axis=0)
    delete_fields = list(radar_c.fields.keys())
    for fl in delete_fields:
        del radar_c.fields[fl]
    mask_dict = {
        'data':blob_mask,
        'units': '',
        'long_name': 'Blob Mask',
        '_FillValue': True,
        'standard_name': 'blob_mask',
    }
    radar_c.add_field('blob_mask', mask_dict, replace_existing=True)

    return radar_c
