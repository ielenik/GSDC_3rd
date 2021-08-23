from re import T

from numpy.core.fromnumeric import reshape
from numpy.lib.function_base import append
from tensorflow.python.ops.variables import Variable
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


class WeightsData(tf.keras.layers.Layer):
    def __init__(self, value, regularizer = None, trainable = True, **kwargs):
        super(WeightsData, self).__init__(**kwargs)
        self.value = value
        self.tr = trainable
        self.regularizer = regularizer

    def build(self, input_shape):
        super(WeightsData, self).build(input_shape)
        self.W = self.add_weight(name='W', shape=self.value.shape, 
                                dtype = tf.float64,
                                initializer=tf.keras.initializers.Constant(self.value),
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

class PsevdoDistancesLayer2(tf.keras.layers.Layer):
    def __init__(self, num_sats, num_epochs, sat_types, sat_poses, psevdo_dist, psevdoweights, **kwargs):
        super(PsevdoDistancesLayer2, self).__init__(**kwargs)
        self.num_sats = num_sats
        self.num_epochs = num_epochs
        self.sat_poses = tf.Variable(sat_poses, trainable=False)
        self.psevdo_dist = tf.Variable(psevdo_dist, trainable=False)
        self.psevdoweights = tf.Variable(psevdoweights, trainable=False)
        self.sat_types = tf.Variable(sat_types, trainable=False)

    def build(self, input_shape):
        super(PsevdoDistancesLayer2, self).build(input_shape)
        self.isrbm_bias = self.add_weight(name='psevdo_isrbm_bias', shape=(8, self.num_epochs), 
                                dtype = tf.float64,
                                initializer=tf.keras.initializers.Constant(np.zeros((8, self.num_epochs))),
                                trainable=True)

    def call(self, inputs):
        distance = tf.linalg.norm(inputs - self.sat_poses, axis = -1)
        isrbm = tf.gather(self.isrbm_bias, self.sat_types)
        isrbm = tf.transpose(isrbm)
        errors = tf.abs(self.psevdoweights*(distance -  self.psevdo_dist - isrbm))
        errors = errors - tf.nn.relu(errors - 1)*0.7

        #return tf.reduce_mean((self.psevdoweights*tf.nn.softsign(tf.abs(errors)/5))), errors
        return tf.reduce_mean(errors), errors

    def compute_output_shape(self, _):
        return (1)

class PsevdoDistancesLayer(tf.keras.layers.Layer):
    def __init__(self, num_sats, num_epochs, base_poses, sat_types, sat_dirs, psevdo_shift, psevdoweights, **kwargs):
        super(PsevdoDistancesLayer, self).__init__(**kwargs)
        self.num_sats = num_sats
        self.num_epochs = num_epochs
        self.base_poses = tf.Variable(base_poses, trainable=False)
        self.sat_types = tf.Variable(sat_types, trainable=False)
        self.sat_dirs = tf.Variable(sat_dirs, trainable=False)
        self.psevdo_shift = tf.Variable(psevdo_shift, trainable=False)
        self.psevdoweights = tf.Variable(psevdoweights, trainable=False)

    def build(self, input_shape):
        super(PsevdoDistancesLayer, self).build(input_shape)
        self.isrbm_bias = self.add_weight(name='psevdo_isrbm_bias', shape=(8, self.num_epochs), 
                                dtype = tf.float64,
                                initializer=tf.keras.initializers.Constant(np.zeros((8, self.num_epochs))),
                                trainable=True)

    def call(self, inputs):
        shiftsnow = tf.reduce_sum( (inputs - self.base_poses)*self.sat_dirs, axis = -1)
        isrbm = tf.gather(self.isrbm_bias, self.sat_types)
        isrbm = tf.transpose(isrbm)
        errors = tf.abs(self.psevdoweights*(shiftsnow -  self.psevdo_shift - isrbm))
        errors = errors - tf.nn.relu(errors - 3)*0.7

        #return tf.reduce_mean((self.psevdoweights*tf.nn.softsign(tf.abs(errors)/5))), errors
        return tf.reduce_mean(errors), errors

    def compute_output_shape(self, _):
        return (1)

class DeltaRangeLayer(tf.keras.layers.Layer):
    def __init__(self, num_sats, num_epochs, sat_directions, sat_deltarange, sat_deltavalid, **kwargs):
        super(DeltaRangeLayer, self).__init__(**kwargs)
        self.num_sats = num_sats
        self.num_epochs = num_epochs
        self.sat_directions = tf.Variable(sat_directions,trainable=False)
        self.sat_deltarange = tf.Variable(sat_deltarange,trainable=False)
        self.sat_deltavalid = tf.Variable(sat_deltavalid,trainable=False)

    def build(self, input_shape):
        super(DeltaRangeLayer, self).build(input_shape)
        self.delta_epoch_bias = self.add_weight(name='delta_epoch_bias', shape=(self.num_epochs-1,1), 
                                dtype = tf.float64,
                                initializer=tf.keras.initializers.Constant(np.zeros((self.num_epochs-1,1))),
                                trainable=True)

    def call(self, inputs):
        shift = inputs[1:] - inputs[:-1]
        scalar = tf.reduce_sum(shift*self.sat_directions, axis = -1)
        errors = (scalar +  self.sat_deltarange - self.delta_epoch_bias)*self.sat_deltavalid
        errors = tf.abs(errors) - tf.nn.relu(tf.abs(errors) - 0.2)*0.7
        #return tf.reduce_mean(tf.nn.softsign(tf.abs(errors))), errors
        return tf.reduce_mean(tf.abs(errors)), errors

    def compute_output_shape(self, _):
        return (1)

def check_deltas_valid(sat_dir, lens):

    n = len(lens)
    if n < 6:
        return False
    num = n*2//3
    if num < 6:
        num = 6
    if num > 8:
        num = 8

    sat_dir = sat_dir.copy()
    sat_dir[:,2] = 1
    for _ in range(250):
        indexes = np.random.choice(n, 3, replace=False)
        mat = sat_dir[indexes]
        vec = lens[indexes]
        try:
            mat = np.linalg.inv(mat)
        except:
            continue

        x_hat = np.dot(mat, vec)
        x_hat = np.reshape(x_hat,(1,3))
        cur_shifts = np.sum(sat_dir*x_hat, axis=1)
        rows = np.abs(cur_shifts - lens)
        if np.sum(rows < 0.1) > num:
            return True
            
    return False
def createGpsPhoneModelFromDataFile(export_data, phone, df_baseline, mat_local):

    times = np.array(export_data[phone]['times'])
    sat_psevdodist      = np.array(export_data[phone]['psevdo'])
    sat_psevdoweights   = np.array(export_data[phone]['psevdo_weight'])
    sat_deltarange = np.array(export_data[phone]['range'])
    sat_deltavalid = np.array(export_data[phone]['range_valid'])
    sat_positions = np.array(export_data[phone]['sats'])
    sat_types_in = np.array(export_data['sat_types'])
    #sat_types_in = sat_types_in[::2].astype(np.int32)
    sat_types = np.zeros((64)).astype(np.int32)
    sat_types[:len(sat_types_in)] = sat_types_in

    num_epochs = len(times)
    baselines = np.zeros((num_epochs, 3))
    num_used_satelites = 64

    for i in range(num_epochs):
        baselines[i] = getValuesAtTimeLiniar(df_baseline['times'], df_baseline['values'], times[i]*1e-6)
        baselines[i] = np.matmul(baselines[i], mat_local)
        for j in range(64):
            sat_positions[i,j] = np.matmul(sat_positions[i,j], mat_local)

    sat_psevdoweights[np.isnan(sat_psevdodist)] = 0
    sat_psevdodist[sat_psevdoweights == 0] = 0

    baselines = np.reshape(baselines,(-1,1,3))
    sat_realdist = np.linalg.norm(sat_positions-baselines, axis = -1)
    sat_realdist[sat_psevdoweights == 0] = 0
    dists = (sat_psevdodist - sat_realdist)
    dists_corr = dists.copy()
    dists_corr[sat_psevdoweights == 0] = 0
    for i in range(num_epochs):
        isbrms = [[],[],[],[],[],[],[],[]]
        for j in range(num_used_satelites):
            if sat_psevdoweights[i,j] == 0:
                continue
            isbrms[sat_types[j]].append(dists_corr[i,j])
        for j in range(8):
            if len(isbrms[j]) == 0:
                isbrms[j] = 0
            else:
                isbrms[j] = np.median(isbrms[j])
        for j in range(num_used_satelites):
            if sat_psevdoweights[i,j] == 0:
                continue
            dists_corr[i,j] -= isbrms[sat_types[j]]
            sat_psevdodist[i,j] -= isbrms[sat_types[j]]

    dists_corr[sat_psevdoweights == 0] = 0
    print(np.sum(abs(dists_corr) > 1000))
    dists_corr = dists_corr*sat_psevdoweights
    loss = np.mean(np.abs(dists_corr[sat_psevdoweights > 0]))
    print("Initial psevdo range loss ", loss)

    sat_directions = sat_positions - baselines
    sat_directions = sat_directions/np.linalg.norm(sat_directions,axis=-1,keepdims=True)

    if False:
        psevdo_layer = PsevdoDistancesLayer2(num_used_satelites,num_epochs,sat_types,sat_positions,sat_psevdodist,sat_psevdoweights)
    else:
        sat_psevdoshift = sat_psevdodist - sat_realdist
        sat_directions[np.isnan(sat_directions)] = 0
        sat_psevdoshift[np.isnan(sat_psevdoshift)] = 0
        sat_psevdoweights[np.isnan(sat_psevdoweights)] = 0
        psevdo_layer = PsevdoDistancesLayer(num_used_satelites,num_epochs,baselines,sat_types,sat_directions,-sat_psevdoshift,sat_psevdoweights)

    sat_directions = sat_directions[1:]
    sat_deltarange = sat_deltarange[1:]
    sat_deltavalid = np.array(sat_deltavalid[1:]).astype(np.float64)

    baselines = baselines[:-1]
    for i in range(num_epochs-1):
        ind = (sat_deltavalid[i]>0)

        '''
        if check_deltas_valid(sat_directions[i,ind],sat_deltarange[i,ind]) == False:
            sat_deltavalid[i] = 0
            sat_deltarange[i] = 0
            continue
        '''
        median_delta = np.median(sat_deltarange[i,ind])
        sat_deltarange[i,ind] -= median_delta

    sat_deltavalid[np.abs(sat_deltarange) > 30] = 0 #
    sat_deltarange[np.abs(sat_deltarange) > 30] = 0 #
    print(np.sum(sat_deltavalid    > 0))
    print(np.sum(sat_psevdoweights > 0))

    sat_deltarange[np.isnan(sat_deltarange)] = 0
    sat_directions[np.isnan(sat_directions)] = 0
    sat_deltavalid[np.isnan(sat_deltavalid)] = 0
    delta_layer = DeltaRangeLayer(num_used_satelites,num_epochs,sat_directions, sat_deltarange, sat_deltavalid)

    model_input = tf.keras.layers.Input((num_epochs,3), dtype=tf.float64)
    positions = tf.reshape(model_input,(num_epochs,1,3))
    psevdo_loss, psevdo_errors = psevdo_layer(positions)
    delta_loss, delta_errors = delta_layer(positions)
    
    gps_phone_model = tf.keras.Model(model_input, [psevdo_loss, delta_loss, delta_errors, psevdo_errors])
    return gps_phone_model, times

def createGpsPhoneModel(df_raw, df_baseline, mat_local, dog, slac):
    LIGHTSPEED = 2.99792458e8

    df_raw['L15'] = '1'
    df_raw.loc[df_raw['CarrierFrequencyHz'] < 1575420030,'L15'] = '5'

    df_raw['Svid_str'] = df_raw['Svid'].apply(str)
    df_raw.loc[df_raw['Svid_str'].str.len() == 1, 'Svid_str'] = '0' + df_raw['Svid_str']
    
    df_raw['Constellation'] ='U'
    df_raw.loc[df_raw['ConstellationType'] == 1, 'Constellation'] = 'G'
    df_raw.loc[df_raw['ConstellationType'] == 3, 'Constellation'] = 'R'
    df_raw.loc[df_raw['ConstellationType'] == 5, 'Constellation'] = 'C'
    df_raw.loc[df_raw['ConstellationType'] == 6, 'Constellation'] = 'E'    
    df_raw = df_raw[df_raw['Constellation'] != 'U'].copy(deep=True) 
    
    df_raw['SvName'] = df_raw['Constellation'] + df_raw['Svid_str']
    df_raw['SvNameType'] = df_raw['Constellation'] + df_raw['Svid_str'] + '_' + df_raw['L15']

    df_raw['NanosSinceGpsEpoch'] = df_raw['TimeNanos'] - df_raw['FullBiasNanos']
    df_raw['PrNanos'] = df_raw['NanosSinceGpsEpoch'] - df_raw['ReceivedSvTimeNanos']
    df_raw['PrNanos'] -= np.floor(df_raw['PrNanos']*1e-9 + 0.02).astype(np.int64) * 1000000000
    df_raw['ReceivedSvTimeNanos'] = df_raw['NanosSinceGpsEpoch'] + df_raw['PrNanos'] # fix sat time
    df_raw['PrM'] = LIGHTSPEED * df_raw['PrNanos'] * 1e-9
    df_raw['PrSigmaM'] = LIGHTSPEED * 1e-9 * df_raw['ReceivedSvTimeUncertaintyNanos']
    df_raw['SAT_FULL_INDEX'] = pd.factorize(df_raw['SvNameType'].tolist())[0]    
    df_raw['ISBRM_INDEX'] = pd.factorize((df_raw['Constellation']+df_raw['L15']).tolist())[0]    

    df_raw['Epoch'] = 0
    df_raw.loc[df_raw['NanosSinceGpsEpoch'] - df_raw['NanosSinceGpsEpoch'].shift() > 10*1e6, 'Epoch'] = 1
    df_raw['Epoch'] = df_raw['Epoch'].cumsum()
    #num_epochs         = df_raw['Epoch'].max() + 1
    #df_raw= df_raw[df_raw['Epoch'] >= num_epochs - 500].copy()
    #df_raw['Epoch'] -= num_epochs - 500

    delta_millis = df_raw['PrNanos'] / 1e6
    where_good_signals = (delta_millis > -20) & (delta_millis < 300)
    df_invalide = df_raw[~where_good_signals]
    print(np.sum(~where_good_signals))
    df_raw = df_raw[where_good_signals].copy()


    
    num_used_satelites = df_raw['SAT_FULL_INDEX'].max() + 1
    num_epochs         = df_raw['Epoch'].max() + 1
    
    slac_local_values = np.ones((len(slac['times']),num_used_satelites))*NaN
    sat_names = []
    sat_types = []
    sat_uniq = df_raw.drop_duplicates(['SAT_FULL_INDEX'],keep='last')
    sat_uniq = sat_uniq.sort_values(['SAT_FULL_INDEX'])
    for _, df in sat_uniq.iterrows():
        sat_num = df['SAT_FULL_INDEX']
        sat_name = df['SvNameType']
        while sat_num > len(sat_names):
            sat_names.append('dummy')
            sat_types.append(7)

        sat_names.append(df['SvName'])
        sat_types.append(df['ISBRM_INDEX'])
        if sat_name in sat_registry:
            slac_local_values[:,sat_num] = slac['values'][:,sat_registry[sat_name]]


    slac_coords = np.matmul(slac['coords'],mat_local)
    def getCorrections(time_nanos, rsat):
        psevdoslac = getValuesAtTime(slac['times'], slac_local_values, time_nanos)
        dist_to_sat = np.linalg.norm(rsat - slac_coords, axis = -1)
        return dist_to_sat - psevdoslac

    sat_positions     = np.ones((num_epochs, num_used_satelites, 3))
    sat_psevdodist    = np.zeros((num_epochs, num_used_satelites))
    sat_psevdovalid   = np.zeros((num_epochs, num_used_satelites))
    sat_psevdoweights = np.zeros((num_epochs, num_used_satelites))

    sat_directions = np.zeros((num_epochs, num_used_satelites, 3))
    sat_deltarange = np.zeros((num_epochs, num_used_satelites))
    sat_deltavalid = np.zeros((num_epochs, num_used_satelites))
    
    epoch_times = np.zeros((num_epochs)).astype(np.int64)

    baselines = np.zeros((num_epochs, 3))
    sat_clock_bias = np.zeros((num_used_satelites))

    for epoch_number, epoch in tqdm(df_raw.groupby(['Epoch'])):
        time_nanos = epoch["NanosSinceGpsEpoch"].to_numpy()
        time_nanos = np.sort(time_nanos)
        time_nanos = time_nanos[len(time_nanos)//2]

        epoch_times[epoch_number] = time_nanos
        baselines[epoch_number] = getValuesAtTimeLiniar(df_baseline['times'], df_baseline['values'], time_nanos*1e-6)
        baselines[epoch_number] = np.matmul(baselines[epoch_number], mat_local)

        #if epoch_number == 800:
        #    deltarange      = 123
        #if epoch_number == 66:
        #    break

        for _,r in epoch.iterrows():
            sat_index = r['SAT_FULL_INDEX']
            sat_name  = r['SvName']
            if sat_names[sat_index] != sat_name:
                print("Error in satelite registry")
            

            ReceivedSvTimeNanos = r['ReceivedSvTimeNanos'] - int(sat_clock_bias[sat_index]*1000000000)
            week = int(ReceivedSvTimeNanos/(7*24*60*60*1000000000))
            tow = ReceivedSvTimeNanos/1000000000 - week*7*24*60*60
            timegp = GPSTime(week,tow)
            obj = dog.get_sat_info(sat_names[sat_index], timegp)
            #obj = (0,0,0),0,0,0 #
            if obj is None:
                continue

            sat_pos, sat_vel, sat_clock_err, sat_clock_drift = obj

            sat_clock_bias[sat_index] = sat_clock_err
            sat_positions[epoch_number,sat_index] = sat_pos
            sat_psevdodist[epoch_number,sat_index] = r['PrM']
            sat_psevdovalid[epoch_number,sat_index] = 1
            sat_psevdoweights[epoch_number,sat_index] = 1/r['PrSigmaM']

            sat_deltarange[epoch_number,sat_index] = r['AccumulatedDeltaRangeMeters']
            if sat_deltarange[epoch_number,sat_index] != 0:
                sat_deltavalid[epoch_number,sat_index] = 1

        sat_positions[epoch_number] = rotate_sat(sat_positions[epoch_number], -(int(sat_clock_bias[sat_index]*1000000000) - sat_psevdodist[epoch_number])) 
        sat_positions[epoch_number] = np.matmul(sat_positions[epoch_number], mat_local)
        corr = getCorrections(time_nanos,sat_positions[epoch_number])
        sat_psevdodist[epoch_number] += corr

    sat_psevdovalid[np.isnan(sat_psevdodist)] = 0
    sat_psevdodist[sat_psevdovalid == 0] = 0

    baselines = np.reshape(baselines,(-1,1,3))
    sat_realdist = np.linalg.norm(sat_positions-baselines, axis = -1)
    dists = sat_psevdovalid*(sat_psevdodist - sat_realdist)
    dists_corr = dists.copy()
    for i in range(num_epochs):
        isbrms = [[],[],[],[],[],[],[],[]]
        for j in range(num_used_satelites):
            if sat_psevdovalid[i,j] == 0:
                continue
            isbrms[sat_types[j]].append(dists_corr[i,j])
        for j in range(8):
            if len(isbrms[j]) == 0:
                isbrms[j] = 0
            else:
                isbrms[j] = np.median(isbrms[j])
        for j in range(num_used_satelites):
            if sat_psevdovalid[i,j] == 0:
                continue
            dists_corr[i,j] -= isbrms[sat_types[j]]
            sat_psevdodist[i,j] -= isbrms[sat_types[j]]


    print(np.sum(abs(dists_corr) > 1000))
    sat_psevdovalid[abs(dists_corr) > 1000] = 0
    sat_psevdoweights = sat_psevdoweights*sat_psevdovalid
    dists_corr = dists_corr*sat_psevdoweights
    loss = np.mean(np.abs(dists_corr[sat_psevdovalid > 0]))
    print("Initial psevdo range loss ", loss)

    sat_directions = sat_positions - baselines
    sat_directions = sat_directions/np.linalg.norm(sat_directions,axis=-1,keepdims=True)

    if False:
        psevdo_layer = PsevdoDistancesLayer2(num_used_satelites,num_epochs,sat_types,sat_positions,sat_psevdodist,sat_psevdoweights)
    else:
        sat_psevdoshift = sat_psevdodist - sat_realdist
        psevdo_layer = PsevdoDistancesLayer(num_used_satelites,num_epochs,baselines,sat_types,sat_directions,-sat_psevdoshift,sat_psevdoweights)

    sat_directions = sat_directions[1:]
    sat_deltarange = sat_deltarange[1:] - sat_deltarange[:-1]
    sat_deltavalid = sat_deltavalid[1:]*sat_deltavalid[:-1]

    baselines = baselines[:-1]
    sat_distanse_first = np.linalg.norm(baselines - sat_positions[:-1],axis=-1)
    sat_distanse_next  = np.linalg.norm(baselines - sat_positions[1:],axis=-1)
    sat_distanse_dif = sat_distanse_next - sat_distanse_first
    sat_deltarange -= sat_distanse_dif



    for i in range(num_epochs-1):
        ind = (sat_deltavalid[i]>0)

        if check_deltas_valid(sat_directions[i,ind],sat_deltarange[i,ind]) == False:
            sat_deltavalid[i] = 0
            sat_deltarange[i] = 0
            continue

        median_delta = np.median(sat_deltarange[i,ind])
        sat_deltarange[i,ind] -= median_delta

    sat_deltavalid[np.abs(sat_deltarange) > 30] = 0 #
    sat_deltarange[np.abs(sat_deltarange) > 30] = 0 #
    print(np.sum(sat_deltavalid > 0))
    print(np.sum(sat_psevdovalid > 0))

    delta_layer = DeltaRangeLayer(num_used_satelites,num_epochs,sat_directions, sat_deltarange, sat_deltavalid)

    model_input = tf.keras.layers.Input((num_epochs,3), dtype=tf.float64)
    positions = tf.reshape(model_input,(num_epochs,1,3))
    psevdo_loss, psevdo_errors = psevdo_layer(positions)
    delta_loss, delta_errors = delta_layer(positions)
    
    gps_phone_model = tf.keras.Model(model_input, [psevdo_loss, delta_loss, delta_errors, psevdo_errors])
    return gps_phone_model, epoch_times


def createTrackModel(start_nanos, end_nanos, df_baseline, mat_local):


    #initialize positions at 10hz from baseline
    tick = 500000000
    start_nanos = start_nanos - 10000000000
    end_nanos = end_nanos + 10000000000
    num_measures = (end_nanos - start_nanos)//tick # ten times a second
    positions = np.zeros((num_measures,3))
    for i in range(num_measures):
        positions[i] = getValuesAtTimeLiniar(df_baseline['times'], df_baseline['values'], (i*tick+start_nanos)*1e-6)
        positions[i] = np.matmul(positions[i], mat_local)

    rover_pos  = WeightsData(positions, None, True)
    pos_bias = tf.Variable(np.zeros((1,3)), trainable=True)

    #model to get position at given times (for training phone models)
    model_input = tf.keras.layers.Input((1), dtype=tf.int64)
    rel_times = model_input - start_nanos
    prev_index   = rel_times//tick
    prev_weight  = ((prev_index+1)*tick - rel_times)/tick

    poses = (rover_pos(prev_index) * prev_weight + rover_pos(prev_index+1) * (1 - prev_weight)) + pos_bias
    #dires = (rover_dir(prev_index) * prev_weight + rover_dir(prev_index+1) * (1 - prev_weight))
    track_model = tf.keras.Model(model_input, poses)

    #model to get all position (for training speed/acs etc)
    dummy_input = tf.keras.layers.Input((1), dtype=tf.int64)
    poses = rover_pos(dummy_input) + pos_bias
    #dires = rover_dir(dummy_input)
    track_model_error = tf.keras.Model(dummy_input, poses)

    return track_model, track_model_error, num_measures, start_nanos, tick
