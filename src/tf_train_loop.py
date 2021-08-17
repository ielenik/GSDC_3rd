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


def get_track_path(folder, track):
    
    phone_glob = next(os.walk(folder+"/"+track))[1]
    print(folder, track, end=' ')
    phones = {}
    phone_names = []
    
    if "train" in folder:
        df_baseline   = pd.read_csv("data/baseline_locations_train.csv")    
    else:
        df_baseline   = pd.read_csv("data/baseline_locations_test.csv")    

    df_baseline = df_baseline[df_baseline['collectionName'] == track]
    df_baseline.rename(columns = {'latDeg':'baseLatDeg', 'lngDeg':'baseLngDeg', 'heightAboveWgs84EllipsoidM':'baseHeightAboveWgs84EllipsoidM'}, inplace = True)    
    df_baseline.set_index('millisSinceGpsEpoch', inplace = True)
    df_baseline = df_baseline[~df_baseline.index.duplicated(keep='first')]
    df_baseline.sort_index(inplace=True)

    for phonepath in phone_glob:
        if "train" in folder:
            truepos      = pd.read_csv(folder+"/" + track + "/" + phonepath + "/ground_truth.csv")
            truepos.set_index('millisSinceGpsEpoch', inplace = True)
            df_baseline = df_baseline.combine_first(truepos)

    baseline_times = []
    baseline_ecef_coords = []
    gt_ecef_coords = []
    for timemili, row in df_baseline.iterrows():
        latbl, lonbl, altbl = float(row['baseLatDeg']),float(row['baseLngDeg']),float(row['baseHeightAboveWgs84EllipsoidM'])
        baseline_times.append(timemili)
        baseline_ecef_coords.append(np.array(pm.geodetic2ecef(latbl,lonbl,altbl, deg = True)))
        latbl, lonbl, altbl = float(row['latDeg']),float(row['lngDeg']),float(row['heightAboveWgs84EllipsoidM'] - 61)
        gt_ecef_coords.append(np.array(pm.geodetic2ecef(latbl,lonbl,altbl, deg = True)))
    #baseline_ecef_coords = gt_ecef_coords.copy()

    mat_local = np.zeros((3,3))
    mat_local[2] = baseline_ecef_coords[0]/np.linalg.norm(baseline_ecef_coords[0], axis = -1)
    mat_local[0] = np.array([0,0,1])
    mat_local[0] = mat_local[0] - mat_local[2]*np.sum(mat_local[2]*mat_local[0])
    mat_local[0] = mat_local[0]/np.linalg.norm(mat_local[0], axis = -1)
    mat_local[1] = np.cross(mat_local[0], mat_local[2])
    mat_local = np.transpose(mat_local)
    #mat_local = np.eye(3)

    gt_ecef_coords = np.array(gt_ecef_coords)
    baseline_times = np.array(baseline_times)
    baseline_ecef_coords = np.array(baseline_ecef_coords)
    gt_ecef_coords = np.matmul(gt_ecef_coords,mat_local)

    timeshift = 3657*24*60*60
    datetimenow = int(baseline_times[0])//1000+timeshift
    datetimenow = datetime.datetime.utcfromtimestamp(datetimenow)
    slac_file = loadSlac(datetimenow)
    slac = myLoadRinexPrevdoIndexed(slac_file)
    slac_coords = np.array(myLoadRinex(slac_file).position)
    slac_times = np.array([r[0] for r in slac])
    slac_values = np.array([r[1] for r in slac])

    
    phone_models = []
    phone_times = []
    constellations = ['GPS', 'GLONASS', 'BEIDOU','GALILEO']
    dog = AstroDog(valid_const=constellations, pull_orbit=True)
    phones = {}
    phone_names = []

    max_time = min_time = 0
    for phonepath in phone_glob:
        phone = phonepath
        phones[phone] = len(phones)
        phone_names.append(phone)
        print(phone, end=' ')

        try:
            df_raw =  pd.read_csv(folder + "/" + track + "/" + phone + "/" + phone + "_raw.csv")    
        except:
            logs   = read_log.gnss_log_to_dataframes(folder + "/" + track + "/" + phone + "/" + phone + "_GnssLog.txt")    
            df_raw = logs['Raw']
            df_raw.to_csv(folder + "/" + track + "/" + phone + "/" + phone + "_raw.csv")    

        model, times = tf_phone_model.createGpsPhoneModel(df_raw,{ 'times':baseline_times, 'values':baseline_ecef_coords},mat_local,dog, { 'times':slac_times, 'values':slac_values, 'coords':slac_coords})
        phone_models.append(model)
        phone_times.append(times)

        if min_time == 0 or min_time > times[0]:
            min_time = times[0]
        if max_time == 0 or max_time < times[-1]:
            max_time = times[-1]

    baseline_ecef_coords  = scipy.signal.medfilt(baseline_ecef_coords, [7,1])
    baseline_ecef_coords += np.random.normal(0.,2.,baseline_ecef_coords.shape)
    model_track, track_model_error, num_measures, start_nanos, time_tick = tf_phone_model.createTrackModel(min_time,max_time, { 'times':baseline_times, 'values':baseline_ecef_coords}, mat_local)

    track_input = np.arange(num_measures)
    track_input = np.reshape(track_input,(-1,1))
    @tf.function
    def train_step(optimizer):
        for _ in range(16):
            with tf.GradientTape(persistent=True) as tape:
                total_loss_psevdo = 0
                total_loss_delta = 0
                for i in range(len(phone_models)):
                    poses = model_track(phone_times[i], training=True)
                    poses = tf.reshape(poses,(1,-1,3))
                    psevdo_loss,delta_loss,delta_dif, psev_error = phone_models[i](poses, training=True)
                    total_loss_psevdo += psevdo_loss/10
                    total_loss_delta += delta_loss*2
                poses = track_model_error(track_input, training=True)
                speed = poses[1:] - poses[:-1]
                accs = speed[1:] - speed[:-1]
                speed_loss_small = tf.reduce_mean(tf.abs(speed)) * 0.00001 + tf.reduce_mean(tf.abs(speed[:,2]))/100
                accs_loss_small = tf.reduce_mean(tf.abs(accs)) /10
                accs_loss_large = tf.reduce_mean(tf.nn.relu(tf.abs(accs) - 1)) *10
                #total_loss = accs_loss_small + accs_loss_large + total_loss_psevdo/10 #+ total_loss_delta
                #total_loss = total_loss_psevdo/10 #+ total_loss_delta
                total_loss = ((accs_loss_small + accs_loss_large + speed_loss_small) + total_loss_psevdo) + total_loss_delta


            grads = tape.gradient(total_loss, model_track.trainable_weights)
            optimizer.apply_gradients(zip(grads, model_track.trainable_weights))        
            grads = tape.gradient(total_loss, track_model_error.trainable_weights)
            optimizer.apply_gradients(zip(grads, track_model_error.trainable_weights))        
            for i in range(len(phone_models)):
                grads = tape.gradient(total_loss, phone_models[i].trainable_weights)
                optimizer.apply_gradients(zip(grads, phone_models[i].trainable_weights))        

            del tape
        return accs_loss_small, accs_loss_large, speed_loss_small, total_loss_psevdo, total_loss_delta, delta_dif, poses
    
    lr = 2.
    #optimizer = keras.optimizers.SGD(learning_rate=100., nesterov=True, momentum=0.5)
    #optimizer = keras.optimizers.Adam(learning_rate=0.5)
    optimizer = keras.optimizers.Adam(learning_rate=1)

    for step in range(32*40):

        
        for _ in range(16*2):
            accs_loss_small, accs_loss_large, speed_loss_small, total_loss_psevdo, total_loss_delta, delta_dif, poses = train_step(optimizer)

        pred_pos = model_track(baseline_times*1000000).numpy()
        poses = poses.numpy()
        

        shift = pred_pos - gt_ecef_coords
        shift = shift - np.mean(shift,axis=0,keepdims=True)
        err3d = np.mean(np.linalg.norm(shift,axis = -1))
        dist_2d = np.linalg.norm(shift[:,:2],axis = -1)
        err2d = np.mean(dist_2d)
        dist_2d = np.sort(dist_2d)
        err50 = dist_2d[len(dist_2d)//2]
        err95 = dist_2d[len(dist_2d)*95//100]

        delta_dif = delta_dif.numpy()
        delta_dif = delta_dif[np.abs(delta_dif) > 0]
        percents_good = np.sum(np.abs(delta_dif) < 0.1)*100/len(delta_dif)


        print( "Training loss at step %d (%.2f,%.2f,%.2f,%.2f,%.4f): %.4f,%.4f (%.2f),%.4f,%.4f,%.4f  lr %.4f" % (step, err3d, err2d, err50, err95, (err50+err95)/2, float(total_loss_psevdo), float(total_loss_delta),percents_good,float(accs_loss_large),float(accs_loss_small), float(speed_loss_small), float(lr)), end='\r')
        if(step % 32 == 0):
            lr *= 0.9
            optimizer.learning_rate = lr
            print()
        
    poses = track_model_error(track_input)
    times = start_nanos + time_tick*track_input
    poses = np.matmul(poses, mat_local.T)
    d = {'nanos': np.reshape(times,(-1)), 'X': poses[:,0], 'Y': poses[:,1], 'Z': poses[:,2]}
    df = pd.DataFrame(data=d)
    df.to_csv(folder + "/" + track + "/track.csv")    
