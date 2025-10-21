import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import pyart
from .util import manage_layout

def bio_classification(radars, sweep, fields,
                       colors = ['red', 'blue'],
                       labels = ['Insect', 'Bird'],
                       file_save=None, title=None,
                       xlim=None, ylim=None, rings=None,
                       layout=None, figsize=None, **kwargs):
    cmap = mcolors.ListedColormap(colors)
    if xlim is None:
        xlim = (-250, 250)
    if ylim is None:
        ylim = (-250, 250)
    if rings is None:
        rings = [10, 50, 100, 150, 200, 250]

    legend_h = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(colors))]
    if title is None:
        title = 'Bird vs. Insect Classification'
    sub_title = _create_subtitle(radars[0], sweep)

    nbplot = len(radars)
    if layout is None:
        nrow, ncol = manage_layout(nbplot)
    else:
        nrow = layout[0]
        ncol = layout[1]

    if figsize is None:
        figsize = (8, 8.5)

    fig = plt.figure(figsize=figsize)

    for p in range(nbplot):
        dct = fields[p]
        if isinstance(dct, (list, np.ndarray)):
            dct = {'field': dct}
        elif isinstance(dct, dict):
            if 'field' not in dct:
                if 'BIO_CLASS' in list(radars[p].fields):
                    dct['field'] = 'BIO_CLASS'
                else:
                    raise KeyError('Key "field" not found.')
        else:
            raise KeyError('Key "field" not found.')

        if 'title' not in dct:
            dct['title'] = dct['field']

        ax = fig.add_subplot(nrow, ncol, p + 1)
        display = pyart.graph.RadarDisplay(radars[p])
        display.plot(dct['field'], sweep=sweep,
                     vmin=0, vmax=len(labels) - 1,
                     cmap=cmap, ax=ax,
                     title=dct['title'],
                     axislabels=('', ''),
                     colorbar_flag=False,
                     **kwargs
                    )
        display.set_limits(xlim, ylim, ax=ax)
        display.plot_range_rings(rings, ax=ax, col='lightgray', ls='-.', lw=0.7)
        ax.axhline(y=0, color='lightgray', linestyle='--', linewidth=0.8)
        ax.axvline(x=0, color='lightgray', linestyle='--', linewidth=0.8)
        ax.legend(handles=legend_h, loc='upper right', handlelength=0.8, fontsize=9)

    fig.supxlabel('East West distance from radar (km)', fontsize=10)
    fig.supylabel('North South distance from radar (km)', fontsize=10)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    # fig.text(0.5, 0.94, sub_title, fontsize=11, horizontalalignment='center')
    # plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.text(0.5, 0.91, sub_title, fontsize=11, horizontalalignment='center')
    plt.tight_layout(rect=[0, 0, 1, 1])
    if file_save is None:
        plt.show()
    else:
        plt.savefig(file_save)
        plt.close(fig)

def _create_subtitle(radar, sweep):
    begin_time = pyart.graph.common.generate_radar_time_sweep(radar, sweep)
    time_str = begin_time.isoformat() + 'Z'
    fixed_angle = radar.fixed_angle['data'][sweep]
    return f'Elevation angle: {fixed_angle:.1f} deg. {time_str}'
