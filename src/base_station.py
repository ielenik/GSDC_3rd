import loader 
import slac
import pickle
import os

from tqdm import tqdm



class SlacBaseStation:
    def __init__(self, filename):
        
        timeshift = 3657*24*60*60
        rinex = loader.myLoadRinex(filename)
        self.position = rinex.position
        self.sat_registry = {}
        self.phases = []

        load_val = [ "L1", "L5" ]
        index = {}
        for val in load_val:
            try:
                base = rinex[val]
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

def getSlacBaseStation(dt):
    filename = slac.loadSlac(dt)
    if os.path.exists(filename+'.slac.pkl'):
        with open(filename+'.slac.pkl', 'rb') as f:
            return pickle.load(f)
    
    result = SlacBaseStation(filename)
    with open(filename+'.slac.pkl', 'wb') as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

    return result
