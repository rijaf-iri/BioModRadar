import pyart

def dealiasing_velocity(radar, vel_field, tex_max_thres=3,
                        wind_size=3, nyq=None):
    vel_tex = pyart.retrieve.calculate_velocity_texture(radar, vel_field=vel_field,
                            wind_size=wind_size, nyq=nyq)
    radar.add_field('velocity_texture', vel_tex, replace_existing=True)

    gatefilter = pyart.filters.GateFilter(radar)
    gatefilter.exclude_above('velocity_texture', tex_max_thres)

    vel_dealiased = pyart.correct.dealias_region_based(radar, vel_field=vel_field,
                            nyquist_vel=nyq, centered=True, gatefilter=gatefilter)
    radar.add_field('velocity_corrected', vel_dealiased, replace_existing=True)

    return radar
