import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.ops.array_ops import zeros
from tensorflow.python.training.tracking import base

import georinex as gr
import pickle
import os
from pathlib import Path
from glob import glob
from numpy.core.defchararray import split
import pandas as pd
import pymap3d as pm
import pytz
from tqdm import tqdm
import numpy as np
import math
import gt_total_score

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
    
folder = 'data/test/'
def make_submission():
    model = tf.keras.models.load_model('mymodel.hdf5')

    folders = next(os.walk(folder[:-1]))[1]
    sub       = pd.read_csv("data/submission_v1.csv")
    base      = pd.read_csv("data/baseline_locations_test.csv")
    '''
    base.set_index('millisSinceGpsEpoch', inplace = True)
    base.sort_index(inplace=True)
    base = base[~base.index.duplicated(keep='first')]
    for i, row in tqdm(sub.iterrows()):
        time = int(row['millisSinceGpsEpoch'])
        idx = base.index.searchsorted(time)
        if idx >= len(base.index):
            idx = len(base.index) - 1

        if idx > 0 and time - base.index[idx-1] < base.index[idx] - time:
            idx -= 1
        timegt = base.index[idx]
        r1 = base.loc[timegt]
        if(abs(time-timegt) > 10):
            print('!!!No values for', time)
        else:
            sub.at[i,'latDeg'] = r1['latDeg']
            sub.at[i,'lngDeg'] = r1['lngDeg']
    '''
    dfs = None
    for f in folders:
        submission      = pd.read_csv(folder+f + "/submission.csv")
        submission['track'] = f
        if dfs is None:
            dfs = submission
        else:
            dfs = dfs.append(submission, ignore_index=True)
    submission = dfs
    submission.set_index('millisSinceGpsEpoch', inplace = True)
    submission.sort_index(inplace=True)
    submission = submission[~submission.index.duplicated(keep='first')]

    ntimes = 0

    indexes = []
    wgs_pred = []
    data = []
    phone = ""
    trackdict = gt_total_score.trackdict
    phonedict = gt_total_score.phonedict
    if "MTV" not in trackdict: 
        trackdict["MTV"] = 0
    for i, row in tqdm(sub.iterrows()):
        if phone != row['phone']:
            phone = str(row['phone'])
            lastpos = []
            lastdir = np.array([ 0., 0. ])
            curtrack = phone.split('_')[0][-5:-2]
            if curtrack not in trackdict:
                curtrack = 'MTV'
            curtrack = trackdict[curtrack]

            t = phone.split('_')[1]
            if t not in phonedict:
                phonedict[t] = len(phonedict)
                print("error")
            curphone = phonedict[t]

            idx_loc = 0
        
        #if "2020-08-13-US-MTV-1" != phone.split('_')[0]:
        #    continue

        
        time = int(row['millisSinceGpsEpoch'])
        idx = submission.index.searchsorted(time)
        if idx >= len(submission.index):
            idx = len(submission.index) - 1

        if idx_loc == 0:
            idx += 1

        timegt = submission.index[idx]
        timegt2 = submission.index[idx-1]
        if idx_loc > 1 and submission.at[timegt, 'track'] != submission.at[timegt2, 'track']:
            idx -= 1

        idx_loc += 1
        timegt = submission.index[idx]
        r1 = submission.loc[timegt]

        timegt2 = submission.index[idx-1]
        r2 = submission.loc[timegt2]
        if abs(timegt - timegt2) < 30000:
            #print('Aproximating', time, row['phone'])
            k1 = (time - timegt2)/(timegt - timegt2)
            lat = r1['latDeg']*k1 + r2['latDeg']*(1-k1)
            lon = r1['lonDeg']*k1 + r2['lonDeg']*(1-k1)
        else:
            ntimes += 1
            print('No values for', time, row['phone'], ntimes)
            continue
        
        wgs_pred.append(np.array([lat,lon]))
    
        dir = lastdir
        if len(lastpos) != 0:
            dir2 = (wgs_pred[-1] - lastpos)
            if np.linalg.norm(dir2) > 1e-7:
                dir = dir2/np.linalg.norm(dir2)
                lastdir = dir

        dt =[dir[0],dir[1]]
        dt.extend(tf.keras.utils.to_categorical(curphone,7))
        dt.extend(tf.keras.utils.to_categorical(curtrack,5))
        dt.extend(wgs_pred[-1])
        data.append(dt)
        lastpos = wgs_pred[-1]
        indexes.append(i)
    
    data1 = np.reshape(data,(-1,16,1))
    data2 = np.reshape(data,(-1,1, 16))
    data = np.reshape(data1*data2,(-1,256))

    sh = model(np.array(data)).numpy()
    print(sh)
    wgs_pred = np.array(wgs_pred) + sh*1e-6
    sub.at[indexes,'latDeg']  = wgs_pred[:,0]
    sub.at[indexes,'lngDeg']  = wgs_pred[:,1]



    sub.reset_index(drop=True, inplace=True)
    sub.to_csv("data/submission.csv", index = False)

    header ='<?xml version="1.0" encoding="UTF-8"?>\n'\
            '<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2" xmlns:kml="http://www.opengis.net/kml/2.2" xmlns:atom="http://www.w3.org/2005/Atom">\n'\
            '<Folder>\n'\
            '    <name>GSDC</name>\n'\
            '    <open>1</open>\n'\
            '    <Document>\n'\
            '        <name>submission.kml</name>\n'\
            '        <Folder id="submission">\n'\
            '            <name>submission</name>\n'
    footer ='		</Folder>\n'\
            '    </Document>\n'\
            '</Folder>\n'\
            '</kml>\n'
    track = '<Placemark id="plot">\n'\
			'	<name>%s</name>\n'\
			'	<Style>\n'\
			'		<LineStyle>\n'\
			'			<color>ff0000ff</color>\n'\
			'			<width>5</width>\n'\
			'		</LineStyle>\n'\
			'	</Style>\n'\
			'	<LineString id="poly_plot">\n'\
			'		<tessellate>1</tessellate>\n'\
			'		<coordinates>\n'\
			'			%s\n'\
			'		</coordinates>\n'\
			'	</LineString>\n'\
			'</Placemark\n>'
    sourceFile = open("data/submission.kml", 'w')
    print(header, file = sourceFile)
    for ph, data in sub.groupby('phone'):
        pts = ''
        for index, raw in data.iterrows():
            pts += str(float(raw['lngDeg']))+','+str(float(raw['latDeg']))+',1 '
        pts = pts[:-1]
        print(track%(ph,pts), file = sourceFile)
    print(footer, file = sourceFile)
    sourceFile.close()

