from laika.lib.coordinates import ecef2geodetic, geodetic2ecef
from laika import AstroDog
from laika.gps_time import GPSTime

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.ops.array_ops import zeros
from tensorflow.python.training.tracking import base

import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import pymap3d as pm
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave
import time
import pandas as pd
import loader
import read_log
import itertools
from coords_tools import *
from matplotlib import pyplot
import datetime
from slac import loadSlac
from loader import *
import tf_phone_model

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
autotune = tf.data.experimental.AUTOTUNE
tf.keras.backend.set_floatx('float64')

def create_model(numepochs, initpos):
    epoch = tf.keras.layers.Input((1), dtype=tf.int32)
    positions = tf.keras.layers.Dense(3,use_bias=False, kernel_initializer=tf.keras.initializers.Constant(initpos), name = 'positions', dtype = tf.float64)
    time_bias = tf.keras.layers.Dense(1,use_bias=False, kernel_initializer=tf.keras.initializers.Zeros(), name = 'time_bias')

    def predict_epoch(epoch_z0, pos_in, numepochs):
        epoch_z0 = tf.one_hot(epoch_z0, numepochs, dtype = tf.float64)
        epoch_z0 = tf.squeeze(epoch_z0, axis = 1)
        epoch_z0 = pos_in(epoch_z0)
        return epoch_z0

    epoch_z0 = predict_epoch(epoch, positions, numepochs)
    bias  = predict_epoch(epoch, time_bias, numepochs)

    base_model = tf.keras.Model(epoch, [epoch_z0, bias])

    def kernel_init(shape, dtype=None, partition_info=None):
        kernel = np.zeros(shape)
        kernel[:,0,0] = np.array([-1,1]).astype(np.float64)
        return kernel
    
    pos = tf.keras.layers.Input((3), dtype=tf.float64)
    pos = tf.transpose(pos)

    derivative = tf.keras.layers.Conv1D(1,2,use_bias=False,kernel_initializer=kernel_init, dtype = tf.float64)
    derivative.trainable = False

    vel = derivative(pos)
    acs = derivative(vel)

    acs = tf.transpose(acs)
    acs = tf.reduce_sum(tf.abs(acs), axis = -1)
    acs_model = tf.keras.Model(pos, acs)

    return base_model, acs_model

class WeightsData(tf.keras.layers.Layer):
    def __init__(self, inshape, initializer,regularizer = None, trainable = True, **kwargs):
        super(WeightsData, self).__init__(**kwargs)
        self.inshape = inshape
        self.initializer = initializer
        self.tr = trainable
        self.regularizer = regularizer

    def build(self, input_shape):
        super(WeightsData, self).build(input_shape)
        self.W = self.add_weight(name='W', shape=self.inshape, 
                                dtype = tf.float64,
                                initializer=self.initializer,
                                regularizer=self.regularizer,
                                trainable=self.tr)

    def call(self, inputs):
        return tf.gather_nd(self.W, inputs)
    def compute_output_shape(self, input_shape):
        return [(input_shape[0], 3)]
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'initializer': self.initializer,
            'inshape': self.inshape,
        })
        return config

def map_raw_derived(df_raw, df_derived):
    # Assume we've loaded a dataframe from _GnssLog.txt for only lines beginning with "Raw", we denote this df_raw. Next, assume we've loaded a dataframe from _derived.csv. We denote this df_derived.

    # Create a new column in df_raw that corresponds to df_derived['millisSinceGpsEpoch']
    df_raw['millisSinceGpsEpoch'] = np.floor((df_raw['TimeNanos'] - df_raw['FullBiasNanos']) / 1000000.0).astype(np.uint64)

    # Change each value in df_derived['millisSinceGpsEpoch'] to be the prior epoch.
    raw_timestamps = df_raw['millisSinceGpsEpoch'].unique()
    derived_timestamps = df_derived['millisSinceGpsEpoch'].unique()

    # The timestamps in derived are one epoch ahead. We need to map each epoch
    # in derived to the prior one (in Raw).
    indexes = np.searchsorted(raw_timestamps, derived_timestamps)
    from_t_to_fix_derived = dict(zip(derived_timestamps, raw_timestamps[indexes-1]))
    df_derived['millisSinceGpsEpoch'] = np.array(list(map(lambda v: from_t_to_fix_derived[v], df_derived['millisSinceGpsEpoch'])))

    # Compute signal_type in df_raw.
    # Map from constellation id to frequencies and signals.
    CONSTEL_FREQ_TABLE = {
        0: {'UNKNOWN': (0, 999999999999)},
        1: {
            'GPS_L1': (1563000000, 1587000000),
            'GPS_L2': (1215000000, 1240000000),
            'GPS_L5': (1164000000, 1189000000)
        },
        3: {
            'GLO_G1': (1593000000, 1610000000),
            'GLO_G2': (1237000000, 1254000000)
        },
        4: {
            'QZS_J1': (1563000000, 1587000000),
            'QZS_J2': (1215000000, 1240000000),
            'QZS_J5': (1164000000, 1189000000)
        },
        5: {
            'BDS_B1C': (1569000000, 1583000000),
            'BDS_B1I': (1553000000, 1568990000),
            'BDS_B2A': (1164000000, 1189000000),
            'BDS_B2B': (1189000000, 1225000000)
        },
        6: {
            'GAL_E1': (1559000000, 1591000000),
            'GAL_E5A': (1164000000, 1189000000),
            'GAL_E5B': (1189000000, 1218000000),
            'GAL_E6': (1258000000, 1300000000)
        },
        7: {
            'IRN_S': (2472000000, 2512000000),
            'IRN_L5': (1164000000, 1189000000)
        },
    }


    def SignalTypeFromConstellationAndFequency(constel, freq_hz):
        freqs = CONSTEL_FREQ_TABLE.get(constel, {})
        for id_freq_range in freqs.items():
            rng = id_freq_range[1]
            if rng[0] <= freq_hz <= rng[1]:
                return id_freq_range[0]
        return 'UNKNOWN'


    signal_types = itertools.chain(*[c.keys()
                                for c in CONSTEL_FREQ_TABLE.values()])
    sig_type_cat = pd.api.types.CategoricalDtype(categories=signal_types)
    df_raw['signalType'] = df_raw.apply(lambda r: SignalTypeFromConstellationAndFequency(
        r.ConstellationType, r.CarrierFrequencyHz), axis=1).astype(sig_type_cat)

    # Fix QZS Svids issue.

    # The SVID of any QZS sat in derived may be changed. Since it may be a many to one relationship, we'll need to adjust the values in Raw.
    new_to_old = {1: (183, 193), 2: (184, 194, 196), 3: (
        187, 189, 197, 199), 4: (185, 195, 200)}
    # Maps original svid to new svid for only ConstellationType=4.
    old_to_new = {}
    for new_svid, old_svids in new_to_old.items():
        for s in old_svids:
            old_to_new[s] = new_svid
    df_raw['svid'] = df_raw.apply(lambda r: old_to_new.get(r.Svid, r.Svid) if r.ConstellationType == 4 else r.Svid, axis=1)

def calc_shift_fromsat2d(sat_dir_in, lens_in):
    index = ~np.isnan(lens_in)
    sat_dir = sat_dir_in[index]
    lens = lens_in[index]

    n = len(lens)
    if n < 6:
        return [NaN, NaN, NaN, NaN], []

    ones = np.ones((n,1))
    sat_dir = np.concatenate((sat_dir,ones), axis = -1)

    num = 2*n//3 
    err = 1e7
    if num < 6:
        num = 6


    res = [NaN, NaN, NaN, NaN]
    err_Res = []
    for _ in range(250):
        indexes = np.random.choice(n, 4, replace=False)
        mat = sat_dir[indexes]
        vec = lens[indexes]
        try:
            mat = np.linalg.inv(mat)
        except:
            continue
        x_hat = np.dot(mat, vec)
        x_hat = np.reshape(x_hat,(1,4))

        cur_shifts = np.sum(sat_dir*x_hat, axis=1)
        rows = np.abs(cur_shifts - lens)
        err_cur = rows
        rows = np.sort(rows)
        rows = rows[:num]
        #rows.append(abs(x_hat[2]))
        if -np.sum(rows < 0.1) + abs(x_hat[0,2]) < err:
            err = -np.sum(rows < 0.1) + abs(x_hat[0,2])
            res = x_hat
            err_Res = err_cur
            if err < -num + 1:
                break

            
    ones = np.ones((64,1))
    sat_dir_copy = np.concatenate((sat_dir_in.copy(),ones), axis = -1)
    cur_shifts = np.sum(sat_dir_copy*res, axis=1)
    rows = np.abs(cur_shifts - lens_in)
    return res, rows

def get_track_path(folder, track):
    phone_glob = next(os.walk(folder+"/"+track))[1]
    print(folder, track, end=' ')
    phones = {}
    phone_names = []
    dfs   = None
    poses = []
    if "train" in folder:
        df_baseline   = pd.read_csv("data/baseline_locations_train.csv")    
    else:
        df_baseline   = pd.read_csv("data/baseline_locations_test.csv")    

    df_baseline = df_baseline[df_baseline['collectionName'] == track]
    df_baseline.rename(columns = {'latDeg':'baseLatDeg', 'lngDeg':'baseLngDeg', 'heightAboveWgs84EllipsoidM':'baseHeightAboveWgs84EllipsoidM'}, inplace = True)    
    df_baseline.set_index('millisSinceGpsEpoch', inplace = True)
    df_baseline = df_baseline[~df_baseline.index.duplicated(keep='first')]
    med_isrbm_all = np.zeros((8,8)).astype(np.float64)

    for phonepath in phone_glob:
        phone = phonepath
        print(phone, end=' ')

        if "train" in folder:
            truepos      = pd.read_csv(folder+"/" + track + "/" + phone + "/ground_truth.csv")
            truepos.set_index('millisSinceGpsEpoch', inplace = True)
            df_baseline = df_baseline.combine_first(truepos)

        phones[phone] = len(phones)
        phone_names.append(phone)

        try:
            df_all =  pd.read_csv(folder + "/" + track + "/" + phone + "/" + phone + "_merged.csv")    
        except:
            df_derived   = pd.read_csv(folder + "/" + track + "/" + phone + "/" + phone + "_derived.csv")    
            str_values = ['collectionName', 'phoneName', 'signalType']
            
            for col in df_derived.columns:
                if col in str_values:
                    continue
                df_derived[col] = pd.to_numeric(df_derived[col])

            logs         = read_log.gnss_log_to_dataframes(folder + "/" + track + "/" + phone + "/" + phone + "_GnssLog.txt")    
            df_raw = logs['Raw']


            #delta_millis = df_derived['millisSinceGpsEpoch'] - df_derived['receivedSvTimeInGpsNanos'] / 1e6
            #where_good_signals = (delta_millis > 0) & (delta_millis < 300)
            #df_derived_good = df_derived[where_good_signals].copy()

            map_raw_derived(df_raw,df_derived)
            #delta_millis = df_derived['millisSinceGpsEpoch'] - df_derived['receivedSvTimeInGpsNanos'] / 1e6
            #where_good_signals = (delta_millis > 0) & (delta_millis < 300)
            #df_derived = df_derived[where_good_signals].copy()
            #df_derived = df_derived.append(df_derived_good, ignore_index=True)
            #df_derived.sort_values(by = 'millisSinceGpsEpoch', inplace = True)
            #print(df_derived.head())
            #print(df_derived_good.head())
            
            
            df_all = pd.merge(df_raw,df_derived)
            df_all.to_csv(folder + "/" + track + "/" + phone + "/" + phone + "_merged.csv")     

        #LIGHTSPEED = 2.99792458e8
        df_all['phoneName'] = phone
        med_isrbm = np.zeros((8)).astype(np.float64)
        
        df_all['stid'] = pd.factorize(df_all['signalType'].tolist())[0]    
        for stid, df in df_all.groupby(['stid']):
            med_isrbm[stid] = df.median()["isrbM"]
        
        df_all["isrbM"] = med_isrbm[df_all['stid']]
        med_isrbm_all[len(phones) - 1,:] = med_isrbm


        df_all["correctedPrM"] = df_all["rawPrM"] + df_all["satClkBiasM"] - df_all["isrbM"] - df_all["ionoDelayM"] - df_all["tropoDelayM"] 
        df_all["correctedPrMTime2"] = df_all["rawPrM"] + df_all["satClkBiasM"] - df_all["isrbM"]
        df_all["correctedPrMSlac"] = df_all["rawPrM"] - df_all["isrbM"]
        LIGHTSPEED = 2.99792458e8
        df_all["correctedReceivedSvTimeInGpsNanos"] = df_all["ReceivedSvTimeNanos"] - (df_all["satClkBiasM"] - df_all["isrbM"])*1e9/LIGHTSPEED

        if dfs is None:
            dfs = df_all
        else:
            dfs = dfs.append(df_all, ignore_index=True)

    constellations = ['GPS', 'GLONASS', 'BEIDOU','GALILEO']

    dog = AstroDog(valid_const=constellations, pull_orbit=True)
    dfs['nanosSinceGpsEpoch'] = dfs['TimeNanos'] - dfs['FullBiasNanos']
    test = dfs['nanosSinceGpsEpoch']
    dfs['prNanos'] = dfs['nanosSinceGpsEpoch'] - dfs['ReceivedSvTimeNanos']
    dfs['fullSeconds'] = np.floor(dfs['prNanos']*1e-9).astype(np.int64)
    dfs['prNanos'] = dfs['prNanos'] - dfs['fullSeconds'] * 1000000000
    test = dfs['prNanos']
    dfs['PrM'] = LIGHTSPEED * dfs['prNanos'] * 1e-9
    test = dfs['PrM']
    dfs['PrSigmaM'] = LIGHTSPEED * 1e-9 * dfs['ReceivedSvTimeUncertaintyNanos']
    dfs["correctedReceivedSvTimeInGpsNanos"] = dfs['nanosSinceGpsEpoch'] + dfs['prNanos'] - (dfs["satClkBiasM"] - dfs["isrbM"])*1e9/LIGHTSPEED
    dfs["correctedPrMTime"] = (dfs["nanosSinceGpsEpoch"] - dfs["correctedReceivedSvTimeInGpsNanos"])*LIGHTSPEED*1e-9

    delta_millis = dfs['prNanos'] / 1e6
    where_good_signals = (delta_millis > 0) & (delta_millis < 300)
    dfs = dfs[where_good_signals]

    df_baseline.sort_index(inplace=True)
    dfs['sfid'] = dfs['signalType']+dfs['svid'].astype(str)
    dfs['sfid'] = pd.factorize(dfs['sfid'].tolist())[0]    
    print()

    timeshift = 3657*24*60*60
    datetimenow = int(df_all.at[0,'millisSinceGpsEpoch'])//1000+timeshift
    datetimenow = datetime.datetime.utcfromtimestamp(datetimenow)
    slac_file = loadSlac(datetimenow)
    slac = myLoadRinexPrevdoIndexed(slac_file)
    slac_coords = np.array(myLoadRinex(slac_file).position)
    slac_times = np.array([r[0] for r in slac])
    slac_old = np.array([r[1] for r in slac])
    slac = np.ones((len(slac_times),64))*NaN
    sat_types = []
    sat_names = []
    mat_local = np.array([[1,0,0],[0,1,0],[0,0,1]])

    
    baseline_times = []
    baseline_ecef_coords = []
    for timenano, row in df_baseline.iterrows():
        latbl, lonbl, altbl = float(row['baseLatDeg']),float(row['baseLngDeg']),float(row['baseHeightAboveWgs84EllipsoidM'])
        baseline_times.append(timenano)
        baseline_ecef_coords.append(np.array(pm.geodetic2ecef(latbl,lonbl,altbl, deg = True)))


    for i in range(64):
        try:
            reg = dfs[dfs.sfid == i].iloc[0]
            num = str(int(reg['svid']))
            if len(num) == 1:
                num = "0"+num
            typ = str(reg['signalType'])
            if typ[:3] == "GPS":
                tpd = "G"
            elif typ[:3] == "GLO":
                tpd = "R"
            elif typ[:3] == "GAL":
                tpd = "E"
            elif typ[:3] == "BDS":
                tpd = "C"
            else:
                tpd = "U"
            
            tpd += num
            sat_names.append(tpd)

            if '5' in typ:
                tpd += "_5"
            else:
                tpd += "_1"
            
            sat_types.append(tpd)
            if tpd in sat_registry:
                ind = sat_registry[tpd]
                slac[:,i] = slac_old[:,ind]
        except:
            break


    def getCorrections(time_nanos, rsat):
        psevdoslac = getValuesAtTime(slac_times, slac, time_nanos)
        dist_to_sat = np.linalg.norm(slac_coords-rsat, axis = -1)
        return dist_to_sat - psevdoslac
    
    num_phones  = len(phones)
    prev_accumrange = np.zeros((num_phones,64)) * NaN
    prev_sat = np.zeros((num_phones,64, 3)) * NaN
    prev_epoch = np.zeros((num_phones)).astype(int)
    rover_last= np.zeros((num_phones, 3)) * NaN
    
    base_poses = []
    base_biases = []
    true_poses = []
    base_times = []

    sat_psevdo_coords       = []
    sat_psevdo_range        = []
    sat_psevdo_epoch        = []
    sat_psevdo_phone    = []
    sat_psevdo_weight   = []
    sat_psevdo_type     = []
    sat_psevdo_type2     = []

    sat_range_vects     = []
    sat_range_change    = []
    sat_range_phone    = []
    sat_range_epoch     = []
    sat_range_prev_epoch   = []
    
    up_vect = []
    mat = np.zeros((3,3))

    bias_last = [ 0,0,0,0,0,0,0]
    epoch_num = 0
    dfs['Epoch'] = 0
    dfs.loc[dfs['nanosSinceGpsEpoch'] - dfs['nanosSinceGpsEpoch'].shift() > 1000000, 'Epoch'] = 1
    dfs['Epoch'] = dfs['Epoch'].cumsum()


    time_bias_range = [[],[],[],[],[]]
    time_bias_delta = [[],[],[],[],[]]
    track_speeds = [[],[],[],[],[]]


    for (time_nanos, phone), epoch in tqdm(dfs.groupby(['nanosSinceGpsEpoch', 'phoneName'])):
        phoneind = phones[phone]
        time = time_nanos*1e-6

        if True:
            #time = time - 10000
            idx = df_baseline.index.searchsorted(time)
            if idx >= len(df_baseline.index):
                idx = len(df_baseline.index) - 1

            if idx > 0 and time - df_baseline.index[idx-1] < df_baseline.index[idx] - time:
                idx -= 1
            timegt = df_baseline.index[idx]
            row = df_baseline.loc[timegt]
            latbl, lonbl, altbl = float(row['baseLatDeg']),float(row['baseLngDeg']),float(row['baseHeightAboveWgs84EllipsoidM'])
            posbl =   np.array(pm.geodetic2ecef(latbl,lonbl,altbl, deg = True))

            if(
                #abs(time-timegt) < 10 and 
                "train" in folder):
                lat, lon, alt = row['latDeg'],row['lngDeg'],row['heightAboveWgs84EllipsoidM']
                roverpos =   np.array(pm.geodetic2ecef(lat,lon,alt-61, deg = True))
                true_poses.append(roverpos)
            else:
                if("train" in folder):
                    true_poses.append([NaN,NaN,NaN])
                else: 
                    true_poses.append(posbl)

            if(len(up_vect) == 0):
                up_vect = posbl/np.linalg.norm(posbl, axis = -1)
                
                mat[2] = posbl/np.linalg.norm(posbl, axis = -1)
                mat[0] = np.array([0,0,1])
                mat[0] = mat[0] - mat[2]*np.sum(mat[2]*mat[0])
                mat[0] = mat[0]/np.linalg.norm(mat[0], axis = -1)
                mat[1] = np.cross(mat[0], mat[2])
                mat = np.transpose(mat)

            base_poses.append(posbl)
            base_times.append(time)
        

        psevdorover = np.array([NaN]*64)
        psevdoslac = np.array([NaN]*64)
        accumrange = np.array([NaN]*64)
        uncert = np.array([NaN]*64)
        timesinm = np.array([NaN]*64)
        signaltype = np.array([[0,0]]*64)
        sats = np.array([[NaN,NaN,NaN]]*64)
        sats2 = np.array([[NaN,NaN,NaN]]*64)
        receivedSvTimeInGpsNanos = np.array([NaN]*64)

        epnp = epoch[['sfid','correctedPrM','AccumulatedDeltaRangeMeters', 'rawPrUncM', 'correctedPrMTime', 'PrM','correctedReceivedSvTimeInGpsNanos', 'xSatPosM','ySatPosM','zSatPosM','stid']].to_numpy()
        satind = epnp[:,0].astype(int)
        sats2[satind]         = epnp[:,-4:-1]
        psevdorover[satind]  = epnp[:,1]
        accumrange[satind]   = epnp[:,2]
        uncert[satind]   = epnp[:,3]
        timesinm[satind]   = epnp[:,4]
        psevdoslac[satind]   = epnp[:,5]
        receivedSvTimeInGpsNanos[satind] = epnp[:,6]
        signaltype[satind,1]   = epnp[:,-1]
        signaltype[satind,0]   = epoch_num

        uncert = 1/uncert
        for i in range(len(sat_names)):
            if np.isnan(receivedSvTimeInGpsNanos[i]):
                continue

            week = int(receivedSvTimeInGpsNanos[i]*1e-9/(7*24*60*60))
            tow = receivedSvTimeInGpsNanos[i]*1e-9 - week*7*24*60*60
            timegp = GPSTime(week,tow)
            obj = dog.get_sat_info(sat_names[i], timegp)
            if obj is None:
                psevdoslac[i] = NaN
                continue
            sat_pos, sat_vel, sat_clock_err, sat_clock_drift = obj
            sats[i] = sat_pos
            psevdoslac[i] -= med_isrbm_all[phoneind, signaltype[i,1]]

            
        sats2 -= sats
        rsat = rotate_sat(sats, timesinm) 
        
        corr = getCorrections(time_nanos,rsat)
        index_corr  = ~np.isnan(corr)
        psevdorover2 = psevdorover.copy()
        psevdorover = psevdoslac + corr

        sat_vect  = rsat-posbl
        sat_vect /= np.linalg.norm(sat_vect, axis = -1, keepdims=True)

        dist_truepos      = np.linalg.norm(rsat-posbl, axis = -1)
        dist_truepos2      = np.linalg.norm(rsat-roverpos, axis = -1)
        dist_truepos_prev = np.linalg.norm(prev_sat[phoneind]-posbl, axis = -1)
        deltarange      = accumrange-prev_accumrange[phoneind] - (dist_truepos - dist_truepos_prev)
        prev_accumrange[phoneind] = accumrange

        index_psev  = ~np.isnan(psevdorover)
        index_delta = ~np.isnan(deltarange)
            
        disterrors = (dist_truepos - psevdorover)[index_psev]
        wdisterrors = sorted(disterrors)
        bias = 0
        if len(disterrors) > 0:
            bias = wdisterrors[len(disterrors)//2]
        disterrors2 = (dist_truepos2 - psevdorover)[index_psev]
        
        wdisterrors2 = sorted(disterrors2)
        bias2 = 0
        if len(disterrors2) > 0:
            bias2 = wdisterrors2[len(disterrors2)//2]
        time_bias_range[phoneind].append(bias2)

        disterrors -= bias
        base_biases.append([bias])
        prev_epoch_num = int(prev_epoch[phoneind])
        prev_epoch[phoneind] = epoch_num
        if len(psevdorover[index_psev]) >= 4:
            sat_psevdo_coords.extend(rsat[index_psev])
            sat_psevdo_range.extend(psevdorover[index_psev])
            sat_psevdo_weight.extend(uncert[index_psev])
            sat_psevdo_epoch.extend([epoch_num]*len(psevdorover[index_psev])) 
            sat_psevdo_phone.extend([phoneind]*len(psevdorover[index_psev])) 
            sat_psevdo_type.extend(signaltype[index_psev])
            sat_psevdo_type2.extend(signaltype[index_psev, 1])

        real_shift = roverpos-rover_last[phoneind]
        dist_changes = np.sum(sat_vect*np.reshape(real_shift,(1,3)), axis = -1)
        dist_changes = np.reshape(dist_changes,(-1))
        dist_changes = dist_changes[~np.isnan(dist_changes)]
        dist_changes = np.sort(dist_changes)
        if len(dist_changes) > 8:
            drop = (len(dist_changes)+2)//3
            dist_changes = dist_changes[drop:-drop]
            time_bias_delta[phoneind].append(np.mean(dist_changes))
        elif len(dist_changes) > 0:
            time_bias_delta[phoneind].append(dist_changes[len(dist_changes)//2])
        else:
            time_bias_delta[phoneind].append(NaN)

        time_bias_range[phoneind][-1] = NaN
        track_speeds[phoneind].append(np.linalg.norm(real_shift))

        if len(deltarange[index_delta]) >= 6:
            shft, err = calc_shift_fromsat2d(sat_vect, deltarange)
            time_bias_range[phoneind][-1] = shft[0,3]
            bias_last[phoneind] = bias - bias_last[phoneind]
            index_delta_last = index_delta
            err[~index_delta] = 1000
            index_delta = np.abs(err) < 0.1
            if len(deltarange[index_delta]) >= 6: #(len(deltarange[index_delta]) >= 8 or len(deltarange[index_delta_last])*3//4 <= len(deltarange[index_delta])):
                sat_range_vects.extend(sat_vect[index_delta_last])
                sat_range_change.extend(deltarange[index_delta_last])
                sat_range_epoch.extend([epoch_num]*len(deltarange[index_delta_last])) 
                sat_range_prev_epoch.extend([prev_epoch_num]*len(deltarange[index_delta_last])) 
                sat_range_phone.extend([phoneind]*len(deltarange[index_delta_last])) 
            else:
                _, err = calc_shift_fromsat2d(sat_vect, deltarange)

        prev_sat[phoneind] = rsat
        bias_last[phoneind] = bias
        rover_last[phoneind] = roverpos
        epoch_num += 1

    max_bias_len = 0
    for i in range(num_phones):
        #time_bias_range[i].append(0)
        bias_range = np.array(time_bias_range[i])
        #bias_range = bias_range[:-1] - bias_range[1:]
        time_bias_range[i] = bias_range
        if len(bias_range) > max_bias_len:
            max_bias_len = len(bias_range)
        time_bias_delta[i] = np.array(time_bias_delta[i])
        track_speeds[i] = np.array(track_speeds[i])
        
        med = np.median(time_bias_delta[i][~np.isnan(time_bias_delta[i])])
        time_bias_delta[i] -= med
        med = np.median(time_bias_range[i][~np.isnan(time_bias_range[i])])
        time_bias_range[i] -= med

    epoch_arrange = np.arange(max_bias_len)
    lbl = []
    for i in range(num_phones):
        lbl.append(phone_names[i] + " GT bias")
        lbl.append(phone_names[i] + " CALC bias")
        lbl.append(phone_names[i] + " speed")
        pl = np.zeros((max_bias_len))*NaN
        pl[:len(time_bias_range[i])] = time_bias_range[i]
        pl[abs(pl) > 5] = NaN
        pyplot.plot(epoch_arrange, pl + i*10)
        pl[:len(time_bias_range[i])] = time_bias_delta[i]
        pl[abs(pl) > 5] = NaN
        pyplot.plot(epoch_arrange, pl + i*10)
        pl[:len(time_bias_range[i])] = track_speeds[i]/10
        pl[abs(pl) > 5] = NaN
        pyplot.plot(epoch_arrange, pl + i*10)
    pyplot.legend(lbl)
    pyplot.show()        




    sat_psevdo_coords   = np.array(sat_psevdo_coords)
    sat_psevdo_range    = np.array(sat_psevdo_range)
    sat_psevdo_epoch    = np.array(sat_psevdo_epoch)
    sat_psevdo_type     = np.array(sat_psevdo_type)
    sat_psevdo_type2     = np.array(sat_psevdo_type2)
    sat_psevdo_weight   = np.array(sat_psevdo_weight)

    base_poses = np.array(base_poses)
    true_poses = np.array(true_poses)

    sat_range_vects     = np.array(sat_range_vects)
    sat_range_change    = np.array(sat_range_change)
    sat_range_epoch     = np.array(sat_range_epoch)
    sat_range_prev_epoch   = np.array(sat_range_prev_epoch)

    base_biases     = np.array(base_biases)
    base_times     = np.array(base_times)

    up_vect = np.reshape(up_vect,(1,3))
    #epoch_input = tf.squeeze(epoch_input)
    #print(epoch_input)
    
    epoch_input = tf.keras.layers.Input((1), dtype=tf.int32)
    st_input = tf.keras.layers.Input((2), dtype=tf.int32)
    st2_input = tf.keras.layers.Input((1), dtype=tf.int32)
    phone_input = tf.keras.layers.Input((1), dtype=tf.int32)
    base_poses = scipy.signal.medfilt(base_poses, [7,1])
    base_poses += np.random.normal(0.,20.,base_poses.shape)

    #positions = WeightsData((epoch_num,3),tf.keras.initializers.Constant(true_poses),None,False)
    positions = WeightsData((epoch_num,3),tf.keras.initializers.Constant(base_poses))
    time_bias = WeightsData((epoch_num,1),tf.keras.initializers.Constant(base_biases), tf.keras.regularizers.L1L2(0.02,0.02))
    time_bias2 = WeightsData((epoch_num,1),tf.keras.initializers.Constant(np.zeros((epoch_num, 1))), tf.keras.regularizers.L1L2(0.02,0.02))
    satids_bias = WeightsData((epoch_num,8, 1),tf.keras.initializers.Constant(np.zeros((epoch_num,8, 1))), tf.keras.regularizers.L1L2(0.02,0.02))
    phone_delta_bias = WeightsData((8, 1),tf.keras.initializers.Constant(np.zeros((8, 1))))
    satids_common_bias = WeightsData((8, 1),tf.keras.initializers.Constant(np.zeros((8, 1))))
    pos_bias = WeightsData((1, 3),tf.keras.initializers.Constant(np.zeros((1, 3))))
    #time_bias_shifts = WeightsData((epoch_num,1),tf.keras.initializers.Constant(np.zeros(epoch_num)))
    poses = positions(epoch_input) + pos_bias([0])
    bias = time_bias(epoch_input)
    bias2 = time_bias2(epoch_input)
    isbrm =  satids_bias(st_input) + satids_common_bias(st2_input)
    #tb_shift = time_bias_shifts(epoch_input)
    base_model = tf.keras.Model([epoch_input,st_input,st2_input, phone_input], [poses, bias, isbrm, bias2+phone_delta_bias(phone_input)])

    def kernel_init(shape, dtype=None, partition_info=None):
        kernel = np.zeros(shape)
        kernel[:,0,0] = np.array([-1,1]).astype(np.float64)
        return kernel
    
    derivative = tf.keras.layers.Conv1D(1,2,use_bias=False,kernel_initializer=kernel_init, dtype = tf.float64)
    
    def kernel_init_epoch(shape, dtype=None, partition_info=None):
        kernel = np.zeros(shape).astype(np.float64)
        kin = np.zeros((num_phones+1)).astype(np.float64)
        kin[0] = -1
        kin[-1] = 1
        kernel[:,0,0] = kin
        return kernel
    
    derivative_epoch = tf.keras.layers.Conv1D(1,num_phones+1,use_bias=False,kernel_initializer=kernel_init_epoch, dtype = tf.float64)

    time_der =  (base_times[1:] - base_times[:-1])/1000
    time_der[time_der == 0] = 1e-7
    time_der =  np.reshape(time_der,(1,-1,1))
    time_der_tn =  tf.convert_to_tensor(time_der, dtype = tf.float64)


    time_der_speed =  (base_times[num_phones:] - base_times[:-num_phones])/1000
    time_der_speed[time_der_speed == 0] = 1e-7
    time_der_speed =  np.reshape(time_der_speed,(1,-1,1))
    time_der_speed_tn =  tf.convert_to_tensor(time_der_speed, dtype = tf.float64)

    # Instantiate an optimizer to train the model.
    lr = 2.
    optimizer = keras.optimizers.Adam(learning_rate=0.1)
    print(len(sat_psevdo_epoch))
    print(len(sat_range_epoch))
    print(epoch_num)

    sat_psevdo_coords_tn = tf.convert_to_tensor(sat_psevdo_coords, dtype=tf.float64)
    sat_psevdo_epoch_tn = tf.convert_to_tensor(sat_psevdo_epoch, dtype=tf.int32)
    sat_psevdo_type_tn = tf.convert_to_tensor(sat_psevdo_type, dtype=tf.int32)
    sat_psevdo_type2_tn = tf.convert_to_tensor(sat_psevdo_type2, dtype=tf.int32)
    sat_psevdo_range_tn = tf.convert_to_tensor(sat_psevdo_range, dtype=tf.float64)
    sat_psevdo_phone_tn = tf.convert_to_tensor(np.zeros((len(sat_psevdo_range),1)), dtype=tf.int32)
    
    epoch_num_tn = tf.convert_to_tensor(np.arange(epoch_num), dtype=tf.int32)
    epoch_num_d_tn = tf.convert_to_tensor(np.zeros((epoch_num,2)), dtype=tf.int32)
    epoch_num_d2_tn = tf.convert_to_tensor(np.zeros((epoch_num,1)), dtype=tf.int32)
    
    sat_range_phone_tn = tf.convert_to_tensor(np.array(sat_range_phone), dtype=tf.int32)
    sat_range_epoch_tn = tf.convert_to_tensor(sat_range_epoch, dtype=tf.int32)
    sat_range_epoch_m1_tn = tf.convert_to_tensor(sat_range_prev_epoch, dtype=tf.int32)
    sat_range_vects_tn = tf.convert_to_tensor(sat_range_vects, dtype=tf.float64)
    sat_dummyid_tn = tf.convert_to_tensor(np.zeros((len(sat_range_epoch),2)), dtype=tf.int32)
    sat_dummyid2_tn = tf.convert_to_tensor(np.zeros((len(sat_range_epoch),1)), dtype=tf.int32)
    sat_range_change_tn = tf.convert_to_tensor(sat_range_change, dtype=tf.float64)

    @tf.function
    def train_step(optimizer):
        for _ in range(16):
            with tf.GradientTape() as tape:
                #pos_for_psevdo = base_model([sat_psevdo_epoch,sat_psevdo_type], training=True)
                pos_for_psevdo = base_model([sat_psevdo_epoch_tn,sat_psevdo_type_tn,sat_psevdo_type2_tn, sat_psevdo_phone_tn], training=True)
                
                psevdo_distance = tf.linalg.norm(pos_for_psevdo[0] -  sat_psevdo_coords_tn, axis = -1) 
                loss_psevdo = sat_psevdo_weight*tf.abs(psevdo_distance - sat_psevdo_range_tn - tf.squeeze(pos_for_psevdo[1]+pos_for_psevdo[2]))
                loss_psevdo = tf.reduce_mean(tf.nn.softsign(loss_psevdo/5))/2
            
                pos_for_range1 = base_model([sat_range_epoch_tn,sat_dummyid_tn, sat_dummyid2_tn, sat_range_phone_tn], training=True)
                pos_for_range2 = base_model([sat_range_epoch_m1_tn,sat_dummyid_tn, sat_dummyid2_tn, sat_range_phone_tn], training=True)
                shifts = tf.reduce_sum((pos_for_range1[0] - pos_for_range2[0]) * sat_range_vects_tn, axis = -1)  + sat_range_change_tn - tf.squeeze(pos_for_range1[1]-pos_for_range2[1]) - tf.squeeze(pos_for_range1[3])
                loss_range = tf.nn.softsign(tf.abs(shifts))
                #loss_range = tf.abs(shifts)
                #print(shifts)
                #print((pos_for_range1[1] - pos_for_range2[1]))
                #print(sat_range_change)

                loss_range = tf.reduce_mean(loss_range)

                pos_for_accel = base_model([epoch_num_tn,epoch_num_d_tn, epoch_num_d2_tn, epoch_num_d2_tn], training=True)

                poses_batch = tf.transpose(pos_for_accel[0])
                poses_batch = tf.expand_dims(poses_batch, axis=-1)
                
                speed = derivative_epoch(poses_batch)/time_der_speed_tn
                if num_phones > 1:
                    speed = tf.pad(speed,[[0,0],[0,num_phones-1], [0,0]])

                shift1 = derivative(poses_batch)
                shift2 = speed*time_der_tn

                shift_loss = tf.reduce_mean(tf.abs(shift1-shift2)) * 0.01

                accel = derivative(speed)
                accel = tf.squeeze(accel)
                accel = tf.transpose(accel)

                bias_batch = tf.transpose(pos_for_accel[1])
                bias_batch = tf.expand_dims(bias_batch, axis=-1)

                loss_der1 = tf.reduce_mean(tf.nn.relu(tf.abs(accel) - 4)) 
                loss_der2 = tf.reduce_mean(tf.abs(accel)) * 0.01

                loss_der =  loss_der1+loss_der2+shift_loss

                total_loss = loss_psevdo*1e-1 #+ loss_range/2 + loss_der * 5 

            grads = tape.gradient(total_loss, base_model.trainable_weights)
            optimizer.apply_gradients(zip(grads, base_model.trainable_weights))        

        return pos_for_accel, shifts, loss_psevdo, loss_range, loss_der, loss_der1, loss_der2, shift_loss, accel, speed
    
    for step in range(32*40):

        for _ in range(16*2):
            pos_for_accel, shifts, loss_psevdo, loss_range, loss_der, loss_der1, loss_der2, shift_loss, accel, speed = train_step(optimizer)
        
        shifts = shifts.numpy()
        sh = len(shifts[np.abs(shifts) < 0.3])*100./len(shifts)

        pos_3d_pred = np.matmul(pos_for_accel[0].numpy(), mat)
        pos_3d_true = np.matmul(true_poses, mat)
        shift2d = (pos_3d_pred-pos_3d_true)[:,:2]
        dist_2d = np.linalg.norm(shift2d, axis = -1)
        loss_true_2d = np.mean(dist_2d)
        corr2d = np.mean(shift2d, axis = 0, keepdims = True)
        shift2d -= corr2d
        dist_2d = np.linalg.norm(shift2d, axis = -1)
        loss_true_2d = np.mean(dist_2d)
        shift3d = (pos_3d_pred-pos_3d_true)
        corr3d = np.mean(shift3d, axis = 0, keepdims = True)
        shift3d -= corr3d
        dist_3d = np.linalg.norm(shift3d, axis = -1)
        loss_true_3d = np.mean(dist_3d)

    
    
        idx_bad = (dist_2d > 10)
        dist_2d = sorted(dist_2d)
        err50 = dist_2d[len(dist_2d)//2]
        err95 = dist_2d[len(dist_2d)*95//100]
        
        print( "Training loss at step %d (%.1f, %.1f, %.1f, %.1f, %.5f): %.4f, %.4f (%.2f), %.4f(%.2f,%.2f,%.2f),  lr %.4f" % (step, float(loss_true_3d), float(loss_true_2d), err50, err95, (err50+err95)/2, float(loss_psevdo), float(loss_range), sh, float(loss_der), float(loss_der1), float(loss_der2), float(shift_loss), float(lr)), end='\r')
        
        if(step % 32 == 0):
            lr *= 0.9
            optimizer.learning_rate = lr
            print()

            if False:
                accel = accel.numpy()
                speed = speed.numpy()

                base_poses_2d = np.matmul(base_poses, mat)
                plt.scatter(pos_3d_pred[:,1], pos_3d_pred[:,0])
                plt.scatter(pos_3d_true[:,1], pos_3d_true[:,0])
                plt.scatter(base_poses_2d[idx_bad,1], base_poses_2d[idx_bad,0] + 1)
                plt.scatter(pos_3d_pred[idx_bad,1], pos_3d_pred[idx_bad,0] + 1)
                plt.scatter(pos_3d_true[idx_bad,1], pos_3d_true[idx_bad,0] + 1)
                plt.show()

    d213123 = {'millisSinceGpsEpoch': base_times}
    df_gt = pd.DataFrame(data=d213123)
    pos_3d_pred = pos_for_accel[0].numpy()
    pos_3d_pred = np.array(pm.ecef2geodetic(pos_3d_pred[:,0],pos_3d_pred[:,1],pos_3d_pred[:,2]))
    df_gt['latDeg'] = pos_3d_pred[0,:]
    df_gt['lonDeg'] = pos_3d_pred[1,:]
    df_gt.to_csv(folder + "/" + track + "/submission.csv")

            