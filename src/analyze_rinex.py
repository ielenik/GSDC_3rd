from slac import loadSlac
from pathlib import Path
from glob import glob
import pandas as pd
import pymap3d as pm
import pytz
from tqdm import tqdm
from loader import *
import numpy as np
import math
import tensorflow as tf


from ftplib import FTP
import ftplib
import datetime
from dateutil.parser import parse
import subprocess
import sys
from glob import glob
from bisect import bisect

from matplotlib import pyplot
from sklearn.linear_model import (
    LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor)
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import scipy
import scipy.optimize as opt
from scipy.signal import medfilt
import tensorflow_position

NaN = float("NaN")

rinex = myLoadRinex('data/train/2020-05-14-US-MTV-1/Pixel4/supplemental/Pixel4_GnssLog.20o')
c1c = rinex['C1C']
l1c = rinex['L1C']
d1c = rinex['D1C']
c5c = rinex['C5X']
l5c = rinex['L5X']
d5c = rinex['D5X']
'''
rinex = myLoadRinex('slac/slac1350.20o')
c1c = rinex['C1']
d1c = rinex['L1']
c5c = rinex['C2']
d5c = rinex['L2']
'''
for i in range(32):
    times_c1c = []
    times_l1c = []
    times_d1c = []
    times_c5c = []
    times_l5c = []
    times_d5c = []
    val_c1c = []
    val_l1c = []
    val_d1c = []
    val_c5c = []
    val_l5c = []
    val_d5c = []

    exists = False
    for ob in tqdm(c1c):
        time = int(ob.time)-timeshift*1000000000
        times_c1c.append(time)
        
        for sat in ob:
            sv = str(sat.sv.data)
            sv = sv.replace(' ','0')
            if sv[0] != 'G':
                continue

            if int(sv[1:]) != i+1:
                continue

            val_c1c.append(float(sat))
            exists = True
            break
    if exists == False:
        continue
    
    for ob in tqdm(d1c):
        time = int(ob.time)-timeshift*1000000000
        times_d1c.append(time)
        
        for sat in ob:
            sv = str(sat.sv.data)
            sv = sv.replace(' ','0')
            if sv[0] != 'G':
                continue

            if int(sv[1:]) != i+1:
                continue

            val_d1c.append(float(sat))
            break
    
    val_c1c.append(val_c1c[-1])
    val_c1c = np.array(val_c1c)
    val_c1c = val_c1c[1:]-val_c1c[:-1]
    val_c1c /= -0.19

    pyplot.plot( times_c1c, val_c1c)
    pyplot.plot( times_d1c, val_d1c)
    
    
    
    for ob in tqdm(c5c):
        time = int(ob.time)-timeshift*1000000000
        times_c5c.append(time)
        
        for sat in ob:
            sv = str(sat.sv.data)
            sv = sv.replace(' ','0')
            if sv[0] != 'G':
                continue

            if int(sv[1:]) != i+1:
                continue

            val_c5c.append(float(sat))
            exists = True
            break
    if exists == False:
        continue
    
    for ob in tqdm(d5c):
        time = int(ob.time)-timeshift*1000000000
        times_d5c.append(time)
        
        for sat in ob:
            sv = str(sat.sv.data)
            sv = sv.replace(' ','0')
            if sv[0] != 'G':
                continue

            if int(sv[1:]) != i+1:
                continue

            val_d5c.append(float(sat))
            break
    
    val_c5c.append(val_c5c[-1])
    val_c5c = np.array(val_c5c)
    val_c5c = val_c5c[1:]-val_c5c[:-1]
    val_c5c /= -0.19

    pyplot.plot( times_c5c, val_c5c)
    pyplot.plot( times_d5c, val_d5c)

    for ob in tqdm(l1c):
        time = int(ob.time)-timeshift*1000000000
        times_l1c.append(time)
        
        for sat in ob:
            sv = str(sat.sv.data)
            sv = sv.replace(' ','0')
            if sv[0] != 'G':
                continue

            if int(sv[1:]) != i+1:
                continue

            if float(sat) != 0:
                val_l1c.append(float(sat))
            else:
                val_l1c.append(NaN)
            break

    if exists == False:
        continue
    
    for ob in tqdm(l5c):
        time = int(ob.time)-timeshift*1000000000
        times_l5c.append(time)
        
        for sat in ob:
            sv = str(sat.sv.data)
            sv = sv.replace(' ','0')
            if sv[0] != 'G':
                continue

            if int(sv[1:]) != i+1:
                continue

            if float(sat) != 0:
                val_l5c.append(float(sat))
            else:
                val_l5c.append(NaN)
            break
    
    val_l1c.append(val_l1c[-1])
    val_l1c = np.array(val_l1c)
    val_l1c = val_l1c[1:]-val_l1c[:-1]
    val_l1c = -val_l1c

    val_l5c.append(val_l5c[-1])
    val_l5c = np.array(val_l5c)
    val_l5c = val_l5c[1:]-val_l5c[:-1]
    val_l5c = -val_l5c

    pyplot.plot( times_l1c, val_l1c)
    pyplot.plot( times_l5c, val_l5c)

    lbl = ["C1C", "D1C", "C5C", "D5C", "L1C", "L5C" ]
    pyplot.legend(lbl)
    pyplot.show()        
