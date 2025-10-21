import os
import re
import copy
import numpy as np
import pandas as pd
import joblib
from .models_fit import compute_fuzzy_scores

# def predict_fuzzy_logic(radar, features, file_model):
#     if not os.path.exists(file_model):
#        raise FileNotFoundError(f'File containing the fuzzy model not found.')

#     model = joblib.load(file_model)

# ## model fitting,
# features = ['REFH_MED', 'PHID_MED', 'RHOV_MED', 'ZDR_MED', 'VV_MED', 'WD_MED']
# ## predict, diffente radar
# fields_dict = {'ref': 'DBZH', 'zdr': 'ZDR', 'rho': 'RHOHV',
#                'phi': 'PHIDP', 'vel': 'VRADH', 'sw': 'WRADH'}
# features = ['DBZH_MED', 'PHIDP_MED', 'RHOHV_MED', 'ZDR_MED', 'VRADH_MED', 'WRADH_MED']
# ## in case features are not ordered
# features = ['ZDR_MED', 'RHOHV_MED', 'DBZH_MED', 'PHIDP_MED', 'WRADH_MED', 'VRADH_MED']

#######

def predict_fuzzy_logic(radar, features, file_stats_bird, file_stats_insect, fields_dict=None):
    if not os.path.exists(file_stats_bird):
       raise FileNotFoundError(f'File containing the bird statistics not found.')
    if not os.path.exists(file_stats_insect):
       raise FileNotFoundError(f'File containing the insect statistics not found.')

    features_stat = copy.deepcopy(features)
    if fields_dict is not None:
        for k, v in fields_dict.items():
            for i in range(len(features_stat)):
                features_stat[i] = re.sub(v, k.upper(), features_stat[i])

    stats_bird = pd.read_csv(file_stats_bird, index_col=0)
    stats_bird = stats_bird[features_stat].T.to_dict(orient='list')
    stats_insect = pd.read_csv(file_stats_insect, index_col=0)
    stats_insect = stats_insect[features_stat].T.to_dict(orient='list')
    data = np.array([radar.fields[f]['data'].filled(np.nan).ravel() for f in features]).T

    scores_bird = compute_fuzzy_scores(data, stats_bird)
    scores_insect = compute_fuzzy_scores(data, stats_insect)

    pred = np.full(data.shape[0], 2, dtype=np.int16)
    pred[scores_bird > scores_insect] = 1
    pred[scores_insect > scores_bird] = 0
    pred = pred.reshape(radar.fields['DR']['data'].shape)
    pred = np.ma.masked_where(pred == 2, pred)

    radar_c = copy.deepcopy(radar)
    radar_fields = list(radar_c.fields)
    delete_fields = [fl for fl in radar_fields if fl not in ['DR', 'DR_CLASS']]
    for fl in delete_fields:
        del radar_c.fields[fl]

    bio_class_dict = {
        'data': pred,
        'units': '',
        'long_name': 'Biological Classification: insect=0, bird=1',
        '_FillValue': -9999,
        'standard_name': 'biological_class',
    }
    radar_c.add_field('BIO_CLASS', bio_class_dict, replace_existing=True)

    return radar_c

def predict_ML_models(radar, features, file_model):
    if not os.path.exists(file_model):
       raise FileNotFoundError(f'File containing the ML model not found.')

    model = joblib.load(file_model)
    data = np.array([radar.fields[f]['data'].filled(np.nan).ravel() for f in features]).T
    if model['scaler'] is not None:
        data = model['scaler'].fit_transform(data)

    pred = np.full(data.shape[0], 2, dtype=np.int16)
    mask = _mask_data_ML(data)
    if np.any(mask):
        pred[mask] = model['model'].predict(data[mask])
    pred = pred.reshape(radar.fields['DR']['data'].shape)
    pred = np.ma.masked_where(pred == 2, pred)

    radar_c = copy.deepcopy(radar)
    radar_fields = list(radar_c.fields)
    delete_fields = [fl for fl in radar_fields if fl not in ['DR', 'DR_CLASS']]
    for fl in delete_fields:
        del radar_c.fields[fl]

    bio_class_dict = {
        'data': pred,
        'units': '',
        'long_name': 'Biological Classification: insect=0, bird=1',
        '_FillValue': -9999,
        'standard_name': 'biological_class',
    }
    radar_c.add_field('BIO_CLASS', bio_class_dict, replace_existing=True)

    return radar_c

def _mask_data_ML(data):
    mask_2d = ~np.isnan(data)
    mask_1d = np.full(data.shape[0], True, dtype=bool)
    for i in range(data.shape[1]):
        mask_1d = np.logical_and(mask_1d, mask_2d[:, i])

    return mask_1d
