import csv
from datetime import datetime

def get_scans_list_weather_biology(csv_file):
    with open(csv_file, 'r') as fl:
        csvreader = csv.DictReader(fl)
        csvdata = []
        for row in csvreader:
            csvdata = csvdata + [row]
    out = []
    for d in csvdata:
        date = datetime.strptime(d['date'], '%Y-%m-%d %H:%M:%S')
        weather = _format_bbox(d['weather'])
        biology = _format_bbox(d['biology'])
        file = d['file']
        out += [{'date': date, 'weather': weather,
                'biology': biology, 'file': file}]
    return out

def get_scans_list_bird_insect(csv_file):
    with open(csv_file, 'r') as fl:
        csvreader = csv.DictReader(fl)
        csvdata = []
        for row in csvreader:
            csvdata = csvdata + [row]
    out = []
    for d in csvdata:
        date = datetime.strptime(d['date'], '%Y-%m-%d %H:%M:%S')
        insect = _format_bbox(d['insect'])
        bird = _format_bbox(d['bird'])
        out += [{'date': date, 'insect': insect, 'bird': bird}]
    return out

def _format_bbox(bbx):
    if bbx == '': return None
    x = bbx.split(';')[1:]
    v = []
    for b in x:
        y = b.split('/')
        mn = y[0].split(',')
        mx = y[1].split(',')
        o = [float(c) for c in mn + mx]
        v += [o]
    return v

# def get_scans_list(csv_file):
#     with open(csv_file, 'r') as fl:
#         csvreader = csv.DictReader(fl)
#         csvdata = []
#         for row in csvreader:
#             csvdata = csvdata + [row]

#     def format_bbox(bbx):
#         if bbx == '': return None
#         x = bbx.split(';')[1:]
#         v = []
#         for b in x:
#             y = b.split('/')
#             mn = y[0].split(',')
#             mx = y[1].split(',')
#             o = [float(c) for c in mn + mx]
#             v += [o]
#         return v

#     out = []
#     for d in csvdata:
#         date = datetime.strptime(d['date'], '%Y-%m-%d %H:%M:%S')
#         insect = format_bbox(d['insect'])
#         bird = format_bbox(d['bird'])
#         out += [{'date': date, 'insect': insect, 'bird': bird}]

#     return out
