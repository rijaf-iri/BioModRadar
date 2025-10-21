import numpy as np
import matplotlib.pyplot as plt
import pyart
from .util import manage_layout

def multiple_RadarDisplay(radar, sweep, fields, file_save=None,
                          xlim=None, ylim=None, rings=None,
                          title=None, figsize=None, **kwargs):
    display = pyart.graph.RadarDisplay(radar)
    if xlim is None:
        xlim = (-250, 250)
    if ylim is None:
        ylim = (-250, 250)
    if rings is None:
        rings = [10, 50, 100, 150, 200, 250]
    if title is None:
        title = _create_title(radar, sweep)

    nbplot = len(fields)
    nrow, ncol = manage_layout(nbplot)
    if figsize is None:
        figsize = (8, 8)

    fig = plt.figure(figsize=figsize)
    for p in range(nbplot):
        dct = fields[p]
        if isinstance(dct, (list, np.ndarray)):
            dct = {'field': dct}
        elif isinstance(dct, dict):
            if 'field' not in dct:
                raise KeyError('Key "field" not found.')
        else:
            raise KeyError('Key "field" not found.')

        if 'title' not in dct:
            dct['title'] = dct['field']
        if 'vmin' not in dct:
            dct['vmin'] = None
        if 'vmax' not in dct:
            dct['vmax'] = None
        if 'cmap' not in dct:
            dct['cmap'] = None

        ax = fig.add_subplot(nrow, ncol, p + 1)
        display.plot(dct['field'], sweep=sweep, cmap=dct['cmap'],
                     vmin=dct['vmin'], vmax=dct['vmax'],
                     title=dct['title'], ax=ax, axislabels=('', ''),
                     colorbar_label='', **kwargs)
        display.set_limits(xlim, ylim, ax=ax)
        display.plot_range_rings(rings, ax=ax, col='lightgray', ls='-.', lw=0.7)
        ax.axhline(y=0, color='lightgray', linestyle='--', linewidth=0.8)
        ax.axvline(x=0, color='lightgray', linestyle='--', linewidth=0.8)

    fig.supxlabel('East West distance from radar (km)', fontsize=10)
    fig.supylabel('North South distance from radar (km)', fontsize=10)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 1])
    if file_save is None:
        plt.show()
    else:
        plt.savefig(file_save)
        plt.close(fig)

def _create_title(radar, sweep):
    begin_time = pyart.graph.common.generate_radar_time_sweep(radar, sweep)
    time_str = begin_time.isoformat() + 'Z'
    fixed_angle = radar.fixed_angle['data'][sweep]
    m = f'{radar.metadata['source']}'
    if m != '':
        m = f'{m}\n'
    d = f'Elevation angle: {fixed_angle:.1f} deg. {time_str}'
    return f'{m}{d}'
