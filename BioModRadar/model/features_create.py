import numpy as np
import pandas as pd

def create_feature_table_bboxs(radar, sweeps, fields, bboxs,
                               drop_na=True, fill_value=np.nan):
    out_df = []
    for sweep in sweeps:
        ranges = radar.range['data']
        radar_s = radar.extract_sweeps([sweep])
        azimuths = radar.get_azimuth(sweep)
        x, y, z = radar.get_gate_x_y_z(sweep)
        x = x/1000.
        y = y/1000.
        mr, ma = np.meshgrid(ranges, azimuths)

        bbx_data = []
        if bboxs is not None:
            df_data = []
            for b in bboxs:
                ix = np.logical_and(x >=b[0], x <= b[2])
                iy = np.logical_and(y >= b[1], y <= b[3])
                ext = np.logical_and(ix, iy)
                field_data = {}
                for field in fields:
                    fl_data = radar_s.fields[field]['data']
                    b_data = fl_data[ext]
                    b_data = b_data.filled(fill_value=fill_value)
                    field_data[field] = np.round(b_data, 4)

                field_data['x'] = np.round(x[ext] * 1000, 2)
                field_data['y'] = np.round(y[ext] * 1000, 2)
                field_data['z'] = np.round(z[ext], 2)
                field_data['ranges'] = np.round(mr[ext])
                field_data['azimuths'] = np.round(ma[ext])
                field_data['sweeps'] = np.full(field_data['x'].shape[0], sweep)
                df = pd.DataFrame(field_data)
                if drop_na:
                    df = df.dropna()
                    df = df.reset_index(drop=True)
                df_data += [df]

            bbx_data += [pd.concat(df_data, axis=0)]
        else:
            bbx_data += [pd.DataFrame(columns=fields)]

        out_df += [pd.concat(bbx_data, axis=0)]

    out_df = pd.concat(out_df, axis=0)
    out_df = out_df.reset_index(drop=True)

    return out_df
