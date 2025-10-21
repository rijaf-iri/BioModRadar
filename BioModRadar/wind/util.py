import pandas as pd
from datetime import datetime, timedelta

def round_bounding_box(bbox, res):
    div = {k: v % res for k, v in bbox.items()}
    bbox['west'] = bbox['west'] - div['west']
    bbox['south'] = bbox['south'] - div['south']
    if div['north'] != 0:
        bbox['north'] = (bbox['north'] - div['north']) + res
    if div['east'] != 0:
        bbox['east'] = (bbox['east'] - div['east']) + res

    return bbox

def cftime2datetime(time):
    return datetime(
                    time.year,
                    time.month,
                    time.day,
                    time.hour,
                    time.minute,
                    time.second
                    )

def npdt64todatetime(time):
    time = pd.Timestamp(time)
    return datetime(
                    time.year,
                    time.month,
                    time.day,
                    time.hour,
                    time.minute,
                    time.second
                    )

def round_to_nearest_hour(time):
    hour = time.replace(second=0,
                        microsecond=0,
                        minute=0,
                        hour=time.hour)
    thour = timedelta(hours=time.minute//30)
    return hour + thour
