
import pickle
import os
import numpy as np
import georinex as gr
from tqdm import tqdm
import pandas as pd

NaN = float("NaN")
timeshift = 3657*24*60*60


if os.path.exists('satreg.pkl'):
    with open('satreg.pkl', 'rb') as f:
        sat_registry = pickle.load(f)
else:
     sat_registry = {}






def myLoadRinex(path):
    if os.path.exists(path+'.pkl'):
        with open(path+'.pkl', 'rb') as f:
            return pickle.load(f)

    d = gr.load(path)
    with open(path+'.pkl', 'wb') as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)
    
    return d

def myLoadRinexPrevdoIndexed(path):
    if os.path.exists(path+'.psevdorange.pkl'):
        with open(path+'.psevdorange.pkl', 'rb') as f:
            return pickle.load(f)

    base_all    = myLoadRinex(path)
    load_val = [ "C5I", "C5X", "C5", "C1C", "C1B", "C1", ]
    index = {}
    sat_distances = []
    for val in load_val:
        try:
            base = base_all[val]
            for ob in tqdm(base):
                time = int(ob.time)-timeshift*1000000000
                if time in index:
                    c1c = index[time]
                else:
                    c1c = [NaN]*(256)
                    cur_sats  = [time, c1c ]
                    sat_distances.append(cur_sats)
                    index[time] = c1c

                for sat in ob:
                    if np.isnan(float(sat)):
                        continue

                    sv = str(sat.sv.data)
                    sv = sv.replace(' ','0')

                    res = float(sat)
                    if res == 0:
                        continue
                    if '5' in val:
                        sv += '_5'
                    else:
                        sv += '_1'
                    if sv not in sat_registry:
                        sat_registry[sv] = len(sat_registry)

                    c1c[sat_registry[sv]] = res
        except Exception as e:
            print(e)
            continue
        
    with open('satreg.pkl', 'wb') as f:
        pickle.dump(sat_registry, f, pickle.HIGHEST_PROTOCOL)
    with open(path+'.psevdorange.pkl', 'wb') as f:
        pickle.dump(sat_distances, f, pickle.HIGHEST_PROTOCOL)

    return sat_distances

def myLoadRinexС1Сindexed(path):
    if os.path.exists(path+'.base.pkl'):
        with open(path+'.base.pkl', 'rb') as f:
            return pickle.load(f)

    SPEED_OF_LIGHT  = 299792458
    GPS_L1_FREQ     = 1575420000
    GPS_L5_FREQ     = 1176450000


    base_all    = myLoadRinex(path)
    '''
    load_val = [ "D5", "D5X", "D1", "D1C", ]
    index_dop = {}
    dopler_shifts = []
    for val in load_val:
        try:
            base = base_all[val]
            for ob in tqdm(base):
                time = int(ob.time)-timeshift*1000000000
                if time in index_dop:
                    dop = index_dop[time]
                else:
                    dop = [NaN]*(256)
                    cur_sats  = [time, dop ]
                    dopler_shifts.append(cur_sats)
                    index_dop[time] = dop
                
                for sat in ob:
                    if np.isnan(float(sat)):
                        continue

                    sv = str(sat.sv.data)
                    sv = sv.replace(' ','0')

                    res = float(sat)
                    if res == 0:
                        continue
                    
                    if '5' in val:
                        sv += '_5'
                    else:
                        sv += '_1'
                    if sv not in sat_registry:
                        sat_registry[sv] = len(sat_registry)
                    dop[sat_registry[sv]] = float(res)
        except Exception as e:
            print(e)
            continue
    '''
    #load_val = [ "C5I", "C5X", "C5", "C1C", "C1B", "C1", ]
    load_val = [ "L5", "L5X", "L1", "L1C", ]
    index = {}
    sat_distances = []
    for val in load_val:
        try:
            base = base_all[val]
            for ob in tqdm(base):
                time = int(ob.time)-timeshift*1000000000
                if time in index:
                    c1c = index[time]
                else:
                    c1c = [NaN]*(256)
                    cur_sats  = [time, c1c ]
                    sat_distances.append(cur_sats)
                    index[time] = c1c

                for sat in ob:
                    if np.isnan(float(sat)):
                        continue

                    sv = str(sat.sv.data)
                    sv = sv.replace(' ','0')

                    res = float(sat)
                    if res == 0:
                        continue
                    if '5' in val:
                        sv += '_5'
                    else:
                        sv += '_1'
                    if sv not in sat_registry:
                        sat_registry[sv] = len(sat_registry)

                    #dop = index_dop[time]
                    #res += dop[sat_registry[sv]]
                    if '5' in val:
                        res = res*SPEED_OF_LIGHT/GPS_L5_FREQ
                    else:
                        res = res*SPEED_OF_LIGHT/GPS_L1_FREQ

                    c1c[sat_registry[sv]] = res
        except Exception as e:
            print(e)
            continue
        
    with open(path+'.base.pkl', 'wb') as f:
        pickle.dump(sat_distances, f, pickle.HIGHEST_PROTOCOL)
    with open('satreg.pkl', 'wb') as f:
        pickle.dump(sat_registry, f, pickle.HIGHEST_PROTOCOL)

    return sat_distances

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
            cur_sats  = [last_time, [[NaN]*10]*(256) ]
            sat_poses.append(cur_sats)

        if row['constellationType'] == 1:
            sv = 'G'
        elif row['constellationType'] == 3:
            sv = 'R'
        elif row['constellationType'] == 6:
            sv = 'E'
        else:
            continue
        sat_num = str(int(row['svid']))
        if len(sat_num) == 1: sat_num = '0' + sat_num


        if row['signalType'][-1] == '1':
            sv = sv + sat_num + '_1'
        else:
            sv = sv + sat_num + '_5'
        
        if sv not in sat_registry:
            sat_registry[sv] = len(sat_registry)
        ind = sat_registry[sv]
        pos = [ float(row['xSatPosM']),float(row['ySatPosM']),float(row['zSatPosM']),
        float(row['satClkBiasM']), float(row['isrbM']),  float(row['ionoDelayM']), float(row['tropoDelayM'])
        , float(row['receivedSvTimeInGpsNanos'])
        , float(row['rawPrM']) 
        , float(row['rawPrUncM'])
        ]  
        #satClkBiasM	satClkDriftMps	rawPrM	rawPrUncM	isrbM	ionoDelayM	tropoDelayM
      
        cur_sats[1][ind] = pos

    with open(path+'.pkl', 'wb') as f:
        pickle.dump(sat_poses, f, pickle.HIGHEST_PROTOCOL)
    with open('satreg.pkl', 'wb') as f:
        pickle.dump(sat_registry, f, pickle.HIGHEST_PROTOCOL)
    
    return sat_poses

