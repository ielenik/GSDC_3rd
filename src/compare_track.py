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
    input_df['dist'] = calc_haversine(input_df['latDeg'].to_numpy(),input_df['lngDeg'].to_numpy(),input_df['ground_truth_latDeg'].to_numpy(),input_df['ground_truth_lngDeg'].to_numpy())
    
def compare_submissions(s1, s2):
    sub1       = pd.read_csv(s1)
    sub2       = pd.read_csv(s2)
    sub1.rename(columns = {'latDeg':'ground_truth_latDeg', 'lngDeg':'ground_truth_lngDeg', 'heightAboveWgs84EllipsoidM':'gtHeightAboveWgs84EllipsoidM'}, inplace = True)    
    sub1 = pd.merge(sub1,sub2, on= 'millisSinceGpsEpoch')
    calc_score(sub1)
    sub1.sort_values('dist', ascending=False, inplace = True)
    print(sub1.head(20))
    sub1.to_csv("data/dif.csv", index = False)
