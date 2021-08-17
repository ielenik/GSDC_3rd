import georinex as gr
import pickle
import os
from pathlib import Path
from glob import glob
import pandas as pd
import pymap3d as pm
import pytz
from datetime import datetime
from tqdm import tqdm
import numpy as np

GPSfromUTC = (datetime(1980,1,6) - datetime(1970,1,1)).total_seconds()

def myLoadRinex(path):
    if os.path.exists(path+'.pkl'):
        with open(path+'.pkl', 'rb') as f:
            return pickle.load(f)

    d = gr.load(path)
    with open(path+'.pkl', 'wb') as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)
    
    return d

NaN = float("NaN")
timeshift = 3657*24*60*60
def myLoadSatPos(path):
    if os.path.exists(path+'.pkl'):
        with open(path+'.pkl', 'rb') as f:
            return pickle.load(f)

    sat_poses = []
    last_time = 0
    cur_sats  = []
    rover2  = pd.read_csv(path)
    for _, row in tqdm(rover2.iterrows()):
        if row['millisSinceGpsEpoch'] != last_time:
            last_time = row['millisSinceGpsEpoch']
            cur_sats  = [last_time, [[NaN]*7]*32 ]
            sat_poses.append(cur_sats)
        if row['constellationType'] != 1:
            continue
        ind = int(row['svid'])
        pos = [ float(row['xSatPosM']),float(row['ySatPosM']),float(row['zSatPosM']),
        float(row['satClkBiasM']), float(row['isrbM']),  float(row['ionoDelayM']), float(row['tropoDelayM'])
        ]  
        #satClkBiasM	satClkDriftMps	rawPrM	rawPrUncM	isrbM	ionoDelayM	tropoDelayM
      
        cur_sats[1][ind] = pos

    with open(path+'.pkl', 'wb') as f:
        pickle.dump(sat_poses, f, pickle.HIGHEST_PROTOCOL)
    
    return sat_poses


def myLoadRinexС1Сindexed(path):
    if os.path.exists(path+'.base.pkl'):
        with open(path+'.base.pkl', 'rb') as f:
            return pickle.load(f)

    base    = myLoadRinex(path)
    base = base["C1C"]
    count = 0
    sat_distances = []
    for ob in tqdm(base):
        time = int(ob.time)-timeshift*1000000000
        '''
        count += 1
        if count > 10:
            break
        '''
        c1c = [NaN]*32
        cur_sats  = [time, c1c ]
        sat_distances.append(cur_sats)
        for sat in ob:
            sv = str(sat.sv.data)
            if(sv[0] != 'G'): continue
            satnum = int(sv[1:])
            c1c[satnum] = float(sat)
    with open(path+'.base.pkl', 'wb') as f:
        pickle.dump(sat_distances, f, pickle.HIGHEST_PROTOCOL)
    
    return sat_distances


base    = myLoadRinexС1Сindexed('data/google-sdc-corrections/osr/rinex/2020-05-14-US-MTV-1.obs')
nav = myLoadRinex('raw_201903280721.nav')


base_coords = np.array(myLoadRinex('data/google-sdc-corrections/osr/rinex/2020-05-14-US-MTV-1.obs').position)

base_times = [r[0] for r in base]
from bisect import bisect

def lerp(a,b,f):
    a = np.array(a)
    b = np.array(b)

    return a*f+b*(1-f)

def getValuesAtTime(index,values,time):
    try:
        i = bisect(index, time)
        if i == 0:
            return lerp(values[i][1], values[i][1], 1)
        if i >= len(values) - 1:
            return lerp(values[-1][1], values[-1][1], 1)

        d = index[i]-index[i-1]
    except:
        return index[i-1]
    return lerp(values[i-1][1], values[i][1], (index[i]-time)/d)


def WGS84_to_ECEF(lat, lon, alt):
    # convert to radians
    rad_lat = lat * (np.pi / 180.0)
    rad_lon = lon * (np.pi / 180.0)
    a    = 6378137.0
    # f is the flattening factor
    finv = 298.257223563
    f = 1 / finv   
    # e is the eccentricity
    e2 = 1 - (1 - f) * (1 - f)    
    # N is the radius of curvature in the prime vertical
    N = a / np.sqrt(1 - e2 * np.sin(rad_lat) * np.sin(rad_lat))
    x = (N + alt) * np.cos(rad_lat) * np.cos(rad_lon)
    y = (N + alt) * np.cos(rad_lat) * np.sin(rad_lon)
    z = (N * (1 - e2) + alt)        * np.sin(rad_lat)
    return x, y, z
    
times = []
difs = []


for time in base_times:
    times.append(time)

    psevdobase = getValuesAtTime(base_times, base, time) 
    dif = psevdobase

    difs.append(dif)

from matplotlib import pyplot
from sklearn.linear_model import (
    LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor)
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import scipy
from scipy.signal import medfilt

times = times[:-5]
difs = difs[:-5]
difs = np.array(difs)
times = np.array(times)
difs[np.abs(difs) > 100000000] = NaN
times = (times - times[0])/(times[-1]- times[0])
lbl = []
for i in range(32):
    index = ~np.isnan(difs[:,i])
    a = difs[:,i][index]
    if len(a) != 0:
        a = np.reshape(difs[:,i][index],(-1,1))
        b = np.reshape(times[index],(-1,1))

#        difs[:,i] -= scipy.signal.medfilt(difs[:,i],131)
        model = make_pipeline(PolynomialFeatures(15), RANSACRegressor())
        model.fit(b, a)
#        difs[:,i] -= np.reshape(model.predict(np.reshape(times,(-1,1))),(-1))
        
#        a = np.reshape(difs[:,i][index],(-1,1))
#        a = difs[:,i][index]
#        model = make_pipeline(PolynomialFeatures(2), TheilSenRegressor())
#        model.fit(b, a)
#        difs[:,i] -= np.reshape(model.predict(np.reshape(times,(-1,1))),(-1))
#        difs[:,i] -= model.predict(np.reshape(times,(-1,1)))
#        inlier_mask = ransac.inlier_mask_
#        outlier_mask = np.logical_not(inlier_mask)

        #difs[:,i] -= np.reshape(model.predict(np.reshape(times,(-1,1))),(-1))
        #difs[:,i] = np.reshape(model.predict(np.reshape(times,(-1,1))),(-1))
        #difs[:,i] = scipy.ndimage.filters.uniform_filter1d(difs[:,i], 21)
#        difs[:,i] = scipy.signal.medfilt(difs[:,i],31)
#        difs[:,i] -= difs[0,i]
        pyplot.plot( times, difs[:,i])
        lbl.append('G'+str(i).zfill(2))
pyplot.ylim((-1, 1))
pyplot.legend(lbl)

pyplot.show()