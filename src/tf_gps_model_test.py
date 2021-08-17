import laika

import glob
import os
import numpy as np
import matplotlib.pyplot as plt

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

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
autotune = tf.data.experimental.AUTOTUNE
tf.keras.backend.set_floatx('float64')


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


def get_track_path(folder, track, submission):
    print(track)
    phone_glob = next(os.walk(folder+"/"+track))[1]

    phones = {}
    dfs   = None
    poses = []
    df_baseline   = pd.read_csv("data/baseline_locations_test.csv")    
    df_baseline = df_baseline[df_baseline['collectionName'] == track]
    if df_baseline.empty:
        df_baseline   = pd.read_csv("data/baseline_locations_train.csv")    
        df_baseline = df_baseline[df_baseline['collectionName'] == track]

    df_baseline.rename(columns = {'latDeg':'baseLatDeg', 'lngDeg':'baseLngDeg', 'heightAboveWgs84EllipsoidM':'baseHeightAboveWgs84EllipsoidM'}, inplace = True)    
    df_baseline.set_index('millisSinceGpsEpoch', inplace = True)
    df_baseline = df_baseline[~df_baseline.index.duplicated(keep='first')]

    for phonepath in phone_glob:
        phone = phonepath #.split('\\')[-1]
        phones[phone] = len(phones)
        print(track, phone)

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
            map_raw_derived(df_raw,df_derived)
            delta_millis = df_derived['millisSinceGpsEpoch'] - df_derived['receivedSvTimeInGpsNanos'] / 1e6
            where_good_signals = (delta_millis > 0) & (delta_millis < 300)
            df_derived = df_derived[where_good_signals].copy()
            

            df_all = pd.merge(df_raw,df_derived)
            df_all.to_csv(folder + "/" + track + "/" + phone + "/" + phone + "_merged.csv")     

        #LIGHTSPEED = 2.99792458e8
        df_all['sfid'] = df_all['signalType']+df_all['svid'].astype(str)
        df_all['sfid'] = pd.factorize(df_all['sfid'].tolist())[0]    
        df_all['stid'] = pd.factorize(df_all['signalType'].tolist())[0]    
        df_all['phoneName'] = phone

        med_isrbm = np.array([ 0,0,0,0,0,0,0,0])
        for _, df in df_all.groupby(['stid']):
            med_isrbm[df['stid']] = df.median()["isrbM"]
        
        df_all["isrbM"] = med_isrbm[df_all['stid']]
        df_all["correctedPrM"] = df_all["rawPrM"] + df_all["satClkBiasM"] - df_all["isrbM"] - df_all["ionoDelayM"] - df_all["tropoDelayM"] 
        df_all["correctedPrMTime"] = df_all["rawPrM"] + df_all["satClkBiasM"] - df_all["isrbM"]

        if dfs is None:
            dfs = df_all
        else:
            dfs = dfs.append(df_all, ignore_index=True)

    num_phones  = len(phones)
    prev_accumrange = np.zeros((num_phones,64)) * NaN
    prev_sat = np.zeros((num_phones,64, 3)) * NaN
    
    base_poses = []
    base_biases = []
    base_times = []

    sat_psevdo_coords   = []
    sat_psevdo_range    = []
    sat_psevdo_epoch    = []
    sat_psevdo_phone    = []
    sat_psevdo_weight   = []
    sat_psevdo_type     = []

    sat_range_vects     = []
    sat_range_change    = []
    sat_range_phone    = []
    sat_range_epoch     = []
    
    up_vect = []
    mat = np.zeros((3,3))
    
    epoch_num = 0
    for (time, phone), epoch in tqdm(dfs.groupby(['millisSinceGpsEpoch', 'phoneName'])):
        phoneind = phones[phone]

        if True:
            idx = df_baseline.index.searchsorted(time)
            if idx >= len(df_baseline.index):
                idx = len(df_baseline.index)-1
            timegt = df_baseline.index[idx]
            row = df_baseline.loc[timegt]
            latbl, lonbl, altbl = float(row['baseLatDeg']),float(row['baseLngDeg']),float(row['baseHeightAboveWgs84EllipsoidM'])
            posbl =   np.array(pm.geodetic2ecef(latbl,lonbl,altbl, deg = True))

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
        accumrange = np.array([NaN]*64)
        uncert = np.array([NaN]*64)
        timesinm = np.array([NaN]*64)
        signaltype = np.array([[0,0]]*64)
        sats = np.array([[NaN,NaN,NaN]]*64)

        epnp = epoch[['sfid','correctedPrM','AccumulatedDeltaRangeMeters', 'rawPrUncM', 'correctedPrMTime', 'xSatPosM','ySatPosM','zSatPosM','stid']].to_numpy()
        satind = epnp[:,0].astype(int)
        sats[satind]         = epnp[:,-4:-1]
        psevdorover[satind]  = epnp[:,1]
        accumrange[satind]   = epnp[:,2]
        uncert[satind]   = epnp[:,3]
        timesinm[satind]   = epnp[:,4]
        signaltype[satind,1]   = epnp[:,-1]
        signaltype[satind,0]   = epoch_num

        uncert = 1/uncert
            

        rsat = rotate_sat(sats, timesinm) 
        '''
        dist_truepos = np.linalg.norm(posbl-sats, axis = -1)
        rsat = rotate_sat(sats, dist_truepos) 
        dist_truepos = np.linalg.norm(posbl-rsat, axis = -1)
        rsat = rotate_sat(sats, dist_truepos) 
        '''

        sat_vect  = rsat-posbl
        sat_vect /= np.linalg.norm(sat_vect, axis = -1, keepdims=True)

        dist_truepos      = np.linalg.norm(rsat-posbl, axis = -1)
        dist_truepos_prev = np.linalg.norm(prev_sat[phoneind]-posbl, axis = -1)
        deltarange      = accumrange-prev_accumrange[phoneind] - (dist_truepos - dist_truepos_prev)
        prev_accumrange[phoneind] = accumrange

        index_psev  = ~np.isnan(psevdorover)
        index_delta = ~np.isnan(deltarange)
            
        disterrors = (dist_truepos - psevdorover)[index_psev]
        disterrors = sorted(disterrors)
        bias = 0
        if len(disterrors) > 0:
            bias = disterrors[len(disterrors)//2]

        base_biases.append([bias])

        if len(psevdorover[index_psev]) > 4:
            sat_psevdo_coords.extend(rsat[index_psev])
            sat_psevdo_range.extend(psevdorover[index_psev])
            sat_psevdo_weight.extend(uncert[index_psev])
            sat_psevdo_epoch.extend([epoch_num]*len(psevdorover[index_psev])) 
            sat_psevdo_phone.extend([phoneind]*len(psevdorover[index_psev])) 
            sat_psevdo_type.extend(signaltype[index_psev])


        if len(deltarange[index_delta]) > 4:
            sat_range_vects.extend(sat_vect[index_delta])
            sat_range_change.extend(deltarange[index_delta])
            sat_range_epoch.extend([epoch_num]*len(deltarange[index_delta])) 
            sat_range_phone.extend([phoneind]*len(deltarange[index_delta])) 

        prev_sat[phoneind] = rsat
        epoch_num += 1

    sat_psevdo_coords   = np.array(sat_psevdo_coords)
    sat_psevdo_range    = np.array(sat_psevdo_range)
    sat_psevdo_epoch    = np.array(sat_psevdo_epoch)
    sat_psevdo_type     = np.array(sat_psevdo_type)
    sat_psevdo_weight   = np.array(sat_psevdo_weight)

    base_poses = np.array(base_poses)
    d213123 = {'millisSinceGpsEpoch': base_times}
    df_gt = pd.DataFrame(data=d213123)
    #submission.set_index('millisSinceGpsEpoch',inplace=True)

    sat_range_vects     = np.array(sat_range_vects)
    sat_range_change    = np.array(sat_range_change)
    sat_range_epoch     = np.array(sat_range_epoch)

    base_biases     = np.array(base_biases)
    base_times     = np.array(base_times)

    
    up_vect = np.reshape(up_vect,(1,3))
    #epoch_input = tf.squeeze(epoch_input)
    #print(epoch_input)
    
    epoch_input = tf.keras.layers.Input((1), dtype=tf.int32)
    st_input = tf.keras.layers.Input((2), dtype=tf.int32)
    #positions = WeightsData((epoch_num,3),tf.keras.initializers.Constant(true_poses))
    positions = WeightsData((epoch_num,3),tf.keras.initializers.Constant(base_poses))
    time_bias = WeightsData((epoch_num,1),tf.keras.initializers.Constant(base_biases))
    satids_bias = WeightsData((epoch_num,8, 1),tf.keras.initializers.Constant(np.zeros((epoch_num,8, 1))), tf.keras.regularizers.L1L2(0.02,0.02))
    #time_bias_shifts = WeightsData((epoch_num,1),tf.keras.initializers.Constant(np.zeros(epoch_num)))
    poses = positions(epoch_input)
    bias = time_bias(epoch_input)
    isbrm =  satids_bias(st_input)
    #tb_shift = time_bias_shifts(epoch_input)
    base_model = tf.keras.Model([epoch_input,st_input], [poses, bias, isbrm])

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
    time_der[time_der==0] = 0.0001
    time_der =  np.reshape(time_der,(1,1,-1,1))
    time_der_tn =  tf.convert_to_tensor(time_der, dtype = tf.float64)
    # Instantiate an optimizer to train the model.

    lr = 2.
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    print(len(sat_psevdo_epoch), len(sat_range_epoch))

    sat_psevdo_epoch_tn = tf.convert_to_tensor(sat_psevdo_epoch, dtype=tf.int32)
    sat_psevdo_type_tn = tf.convert_to_tensor(sat_psevdo_type, dtype=tf.int32)
    sat_psevdo_range_tn = tf.convert_to_tensor(sat_psevdo_range, dtype=tf.float64)
    sat_range_epoch_tn = tf.convert_to_tensor(sat_range_epoch, dtype=tf.int32)
    sat_range_epoch_m1_tn = tf.convert_to_tensor(sat_range_prev_epoch, dtype=tf.int32)
    sat_dummyid_tn = tf.convert_to_tensor(np.zeros((len(sat_range_epoch),2)), dtype=tf.int32)
    sat_range_change_tn = tf.convert_to_tensor(sat_range_change, dtype=tf.float64)
    epoch_num_tn = tf.convert_to_tensor(np.arange(epoch_num), dtype=tf.int32)
    epoch_num_d_tn = tf.convert_to_tensor(np.zeros((epoch_num,2)), dtype=tf.int32)

    @tf.function
    def train_step(optimizer):
        for _ in range(8):
            with tf.GradientTape() as tape:
                #pos_for_psevdo = base_model([sat_psevdo_epoch,sat_psevdo_type], training=True)
                pos_for_psevdo = base_model([sat_psevdo_epoch_tn,sat_psevdo_type_tn], training=True)
                
                psevdo_distance = tf.linalg.norm(pos_for_psevdo[0] -  sat_psevdo_coords, axis = -1) 
                loss_psevdo = sat_psevdo_weight*tf.abs(psevdo_distance - sat_psevdo_range_tn - tf.squeeze(pos_for_psevdo[1]+pos_for_psevdo[2]))
                loss_psevdo = tf.reduce_mean(tf.nn.softsign(loss_psevdo/10))
            
                pos_for_range1 = base_model([sat_range_epoch_tn,sat_dummyid_tn], training=True)
                pos_for_range2 = base_model([sat_range_epoch_m1_tn,sat_dummyid_tn], training=True)
                shifts = tf.reduce_sum((pos_for_range1[0] - pos_for_range2[0]) * sat_range_vects, axis = -1)  + sat_range_change_tn + tf.squeeze(pos_for_range1[1]-pos_for_range2[1])
                loss_range = tf.nn.softsign(tf.abs(shifts))
                #loss_range = tf.abs(shifts)
                #print(shifts)
                #print((pos_for_range1[1] - pos_for_range2[1]))
                #print(sat_range_change)

                loss_range = tf.reduce_mean(loss_range)

                pos_for_accel = base_model([epoch_num_tn,epoch_num_d_tn], training=True)

                poses_batch = tf.transpose(pos_for_accel[0])
                poses_batch = tf.expand_dims(poses_batch, axis=-1)
                speed = derivative(poses_batch)/time_der_tn
                accel = derivative(speed)
                accel = tf.squeeze(accel)
                accel = tf.transpose(accel)

                bias_batch = tf.transpose(pos_for_accel[1])
                bias_batch = tf.expand_dims(bias_batch, axis=-1)

                loss_der1 = tf.reduce_mean(tf.nn.relu(tf.abs(accel) - 4)) 
                loss_der2 = tf.reduce_mean(tf.abs(accel)) * 0.01

                loss_der =  loss_der1+loss_der2

                total_loss = loss_psevdo + loss_range + loss_der
                #total_loss = loss_range

            grads = tape.gradient(total_loss, base_model.trainable_weights)
            optimizer.apply_gradients(zip(grads, base_model.trainable_weights))        

        return pos_for_accel, shifts, loss_psevdo, loss_range, loss_der, loss_der1, loss_der2
    
    for step in range(32*50):

        for _ in range(32):
            pos_for_accel, shifts, loss_psevdo, loss_range, loss_der, loss_der1, loss_der2 = train_step(optimizer)
        
        shifts = shifts.numpy()
        sh = len(shifts[np.abs(shifts) < 0.3])*100./len(shifts)
        
        print( "Training loss at step %d: %.4f, %.4f (%.2f), %.4f(%.2f,%.2f),  lr %.4f" % (step, float(loss_psevdo), float(loss_range), sh, float(loss_der), float(loss_der1), float(loss_der2), float(lr)), end = '\r')
        if(step % 32 == 0):
            print()
            pos_3d_pred = pos_for_accel[0].numpy()
            pos_3d_pred = np.array(pm.ecef2geodetic(pos_3d_pred[:,0],pos_3d_pred[:,1],pos_3d_pred[:,2]))
            df_gt['latDeg'] = pos_3d_pred[0,:]
            df_gt['lonDeg'] = pos_3d_pred[1,:]
            df_gt.to_csv((folder + "/" + track + "/submission.csv")     )
            #submission.update(df_gt)
            lr *= 0.9
            optimizer.learning_rate = lr

