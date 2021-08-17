from loader import NaN
from tqdm import tqdm
import datetime
import pickle
import os
import numpy as np

def get_float(str):
    str = str.strip()
    if len(str) == 0:
        return 0
    return float(str)

def load_google_rinex(filename, sat_registry):
    with open(filename, 'r') as file:
        lines = file.readlines()

    SPEED_OF_LIGHT  = 299792458
    GPS_L1_FREQ     = 1575420000
    GPS_L5_FREQ     = 1176450000

    gpsstart = datetime.datetime(1980, 1, 6)

    times = []
    obs = []
    
    for l in tqdm(lines):
        if l[0] == '>':
            dtstr = l.split(' ')
            dt = datetime.datetime(year=int(dtstr[1]), month=int(dtstr[2]), day=int(dtstr[3]), hour = int(dtstr[4]), minute=int(dtstr[5]))
            seconds = (dt-gpsstart).total_seconds() + float(dtstr[6])
            times.append(seconds*1000000000)
            curobs = np.full((256, 2), NaN)
            obs.append(curobs)
            continue
        if len(times) == 0:
            continue    

        svid = l[:3]
        l1ph = l[20:33]
        l1sg = l[33:35]        
        l5ph = l[84:97]
        l5sg = l[97:99]   

        mesure = get_float(l1ph)
        if  mesure != 0:
            sv = svid + '_1'
            if sv not in sat_registry:
                sat_registry[sv] = len(sat_registry)

            mesure = mesure*SPEED_OF_LIGHT/GPS_L1_FREQ
            curobs[sat_registry[sv], 0] = mesure
            curobs[sat_registry[sv], 1] = int(l1sg)

        mesure = get_float(l5ph)
        if  mesure != 0:
            sv = svid + '_5'
            if sv not in sat_registry:
                sat_registry[sv] = len(sat_registry)

            mesure = mesure*SPEED_OF_LIGHT/GPS_L5_FREQ
            curobs[sat_registry[sv], 0] = mesure
            curobs[sat_registry[sv], 1] = int(l5sg)

    return times, obs