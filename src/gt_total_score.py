import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras import regularizers
from tensorflow.python.ops.array_ops import zeros
from tensorflow.python.training.tracking import base

autotune = tf.data.experimental.AUTOTUNE
tf.keras.backend.set_floatx('float64')

import georinex as gr
import pickle
import os
from pathlib import Path
from glob import glob
from numpy.core.defchararray import split
import pandas as pd
import pymap3d as pm
import pytz
from tensorflow.python.ops.numpy_ops import np_arrays
from tqdm import tqdm
import numpy as np
import math
import matplotlib.pyplot as plt


from ftplib import FTP
import ftplib
import datetime
from dateutil.parser import parse

def calc_haversine(lat1, lon1, lat2, lon2):
  """Calculates the great circle distance between two points
  on the earth. Inputs are array-like and specified in decimal degrees.
  """
  RADIUS = 6_367_000
  lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
  dlat = lat2 - lat1
  dlon = lon2 - lon1
  a = np.sin(dlat/2)**2 + \
      np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
  dist = 2 * RADIUS * np.arcsin(a**0.5)
  return dist

def calc_score(input_df: pd.DataFrame):
    res = calc_haversine(input_df['latDeg'].to_numpy(),input_df['lonDeg'].to_numpy(),input_df['ground_truth_latDeg'].to_numpy(),input_df['ground_truth_lngDeg'].to_numpy())
    p_50 = np.percentile(res, 50)
    p_95 = np.percentile(res, 95)
    return (p_50+p_95)/2
    
folder = 'data/train/'
trackdict = {}
phonedict = {}

def print_gt_score():
    
    folders = next(os.walk(folder[:-1]))[1]
    
    socores = []
    
    
    wgs_true = []
    wgs_pred = []
    data     = []
    if False:
        with open('values_x.pkl', 'rb') as f:
            data = pickle.load(f)
        with open('values_y.pkl', 'rb') as f:
            difwgs = pickle.load(f)
    else:
        model = tf.keras.models.load_model('mymodel.hdf5')
        for f in folders:

            #if 'SJC' not in f:
            #    continue

            try:
                submission      = pd.read_csv(folder+f + "/submission.csv")
                submission.set_index('millisSinceGpsEpoch', inplace = True)
                submission.sort_index(inplace=True)
                submission = submission[~submission.index.duplicated(keep='first')]
            except:
                break

            tracks = next(os.walk(folder+f))[1]
            curtrack = f[-5:-2]
            if curtrack not in trackdict:
                trackdict[curtrack] = len(trackdict)
            curtrack = trackdict[curtrack]
            for t in tracks:
                if t not in phonedict:
                    phonedict[t] = len(phonedict)
                curphone = phonedict[t]

                truepos      = pd.read_csv(folder+f + "/" + t + "/ground_truth.csv")
                lastpos = []
                lastdir = np.array([ 0., 0. ])
                
                count = 0
                for _, row in truepos.iterrows():
                    lat, lon, alt = row['latDeg'],row['lngDeg'],row['heightAboveWgs84EllipsoidM']
                    time = int(row['millisSinceGpsEpoch'])
                    idx = submission.index.searchsorted(time)
                    if idx >= len(submission.index):
                        idx = len(submission.index) - 1

                    if idx > 0 and time - submission.index[idx-1] < submission.index[idx] - time and time - submission.index[idx-1] < 10:
                        idx -= 1
                    if idx == 0:
                        idx = 1

                    timegt = submission.index[idx]
                    r1 = submission.loc[timegt]
                    timegt2 = submission.index[idx-1]
                    r2 = submission.loc[timegt2]

                    if abs(timegt - timegt2) < 10000:
                        #print('Aproximating', time, row['phone'])
                        k1 = (time - timegt2)/(timegt - timegt2)
                        latr = r1['latDeg']*k1 + r2['latDeg']*(1-k1)
                        lonr = r1['lonDeg']*k1 + r2['lonDeg']*(1-k1)
                    else:
                        print('No values for', time, row['phone'])
                        continue

                    count += 1
                    wgs_pred.append(np.array([latr,lonr]))
                    wgs_true.append(np.array([lat,lon]))

                    dir = lastdir
                    if len(lastpos) != 0:
                        dir2 = (wgs_pred[-1] - lastpos)
                        if np.linalg.norm(dir2) > 1e-6:
                            dir = dir2/np.linalg.norm(dir2)
                            lastdir = dir
                    else:
                        scale_lat = calc_haversine(wgs_pred[-1][0], wgs_pred[-1][1], wgs_pred[-1][0]+0.00001, wgs_pred[-1][1])/0.00001
                        scale_lon = calc_haversine(wgs_pred[-1][0], wgs_pred[-1][1], wgs_pred[-1][0], wgs_pred[-1][1]+0.00001)/0.00001

                    dt =[dir[0],dir[1]]
                    dt.extend(tf.keras.utils.to_categorical(curphone,7))
                    dt.extend(tf.keras.utils.to_categorical(curtrack,5))
                    dt.extend((wgs_pred[-1] - np.array([37.41661260440319,-122.08173723432392]))*100)
                    data.append(dt)
                    lastpos = wgs_pred[-1]

                d =np.array(data[-count:])
                sh = model(d).numpy()*1e-6
                wgs_pred2 = np.array(wgs_pred[-count:]) + sh
                wgs_true2 = np.array(wgs_true[-count:])
                wgs_shift = wgs_true2-wgs_pred2
                wgs_shift[:,0] *= scale_lat
                wgs_shift[:,1] *= scale_lon
                dirs = np.array(data)[-count:,:2]
                dirs2 = dirs.copy()
                dirs2[:,0] = -dirs[:,1]
                dirs2[:,1] =  dirs[:,0]
                wgs_shift_dir = np.array([np.sum(dirs*wgs_shift,axis=-1), np.sum(dirs2*wgs_shift,axis=-1)])
                
                ind = np.logical_and(np.linalg.norm(wgs_shift_dir, axis = 0) < 5, np.linalg.norm(wgs_shift_dir, axis = 0) > 0)
                heatmap, xedges, yedges = np.histogram2d(wgs_shift_dir[1,ind], wgs_shift_dir[0,ind], bins=100)
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                #plt.figure()
                #plt.imshow(heatmap.T, extent=extent, origin='lower')

                ind = np.logical_and(np.linalg.norm(wgs_shift, axis = 1) < 5, np.linalg.norm(wgs_shift, axis = 1) > 0)
                heatmap, xedges, yedges = np.histogram2d(wgs_shift[ind,1], wgs_shift[ind,0], bins=100)
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                #plt.figure()
                #plt.imshow(heatmap.T, extent=extent, origin='lower')
                
                err = calc_haversine(wgs_pred2[:,0], wgs_pred2[:,1], wgs_true2[:,0], wgs_true2[:,1])
                err = (np.percentile(err,50)+np.percentile(err,95))/2
                socores.append(err)
                print(f,t, err)

            #plt.show()
        print("total", np.mean(socores))
        data = np.array(data).astype(np.float64)
        difwgs = (np.array(wgs_true).astype(np.float64) - np.array(wgs_pred).astype(np.float64))*1e6

        with open('values_x.pkl', 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        with open('values_y.pkl', 'wb') as f:
            pickle.dump(difwgs, f, pickle.HIGHEST_PROTOCOL)

    inp = tf.keras.layers.Input((16))
    x = inp
    #x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(8, activation='relu')(x)
    x = tf.keras.layers.Dense(8, activation='relu')(x)
    #x = tf.keras.layers.Dense(8, activation='relu')(x)
    #x = tf.keras.layers.BatchNormalization()(inp)
    #x = tf.keras.layers.Dense(8, activation='relu')(x)
    #x = tf.keras.layers.Dense(8, activation='relu')(x)
    x = tf.keras.layers.Dense(2, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-4, l2=1e-8))(x)
    model = tf.keras.Model(inp,x)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.1), loss = 'mae')
    model.fit(data,difwgs, shuffle=True, batch_size = 4096, epochs = 256, callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', verbose=1,  patience=25)] )
    model.save('mymodel.hdf5')


    ind = np.linalg.norm(difwgs, axis = -1) < 20
    heatmap, xedges, yedges = np.histogram2d(difwgs[ind,1], difwgs[ind,0], bins=100)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()

    pred = model(data).numpy()
    print(pred)
    difwgs -= pred

    ind = np.linalg.norm(difwgs, axis = -1) < 20
    heatmap, xedges, yedges = np.histogram2d(difwgs[ind,1], difwgs[ind,0], bins=100)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()
