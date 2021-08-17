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
import phone_track
import height_map
import tf_gps_model
import tf_gps_model_test
import gt_total_score
import make_submision
import compare_track
import tf_train_loop


def main():
    tf_train_loop.get_track_path('data/train','2021-04-22-US-SJC-1')

    subset = 'train'
    tests = next(os.walk('data/'+subset))[1]
    for g in tests:
        tf_train_loop.get_track_path('data/'+subset,g)


    #gt_total_score.print_gt_score()
    #return 
    #tf_gps_model.get_track_path('data/train','2020-07-17-US-MTV-2')
    #tf_train_loop.get_track_path('data/train','2020-07-17-US-MTV-2')
    #tf_train_loop.get_track_path('data/train','2021-04-22-US-SJC-1')
    #tf_train_loop.get_track_path('data/train','2020-05-14-US-MTV-1')
    #tf_train_loop.get_track_path('data/train','2021-04-15-US-MTV-1')
    #return


    gt_total_score.print_gt_score()
    make_submision.make_submission()
    compare_track.compare_submissions("data/submission.csv", "data/submission_ref.csv")
    return

    submission = pd.read_csv("data/sample_submission.csv")    
    folder = 'data/train/2020-05-14-US-MTV-1/'
    phonename = 'Pixel4'
    #phonename = 'Pixel4XLModded'
    folder = folder + phonename + '/'
    #folder = 'data/train/2020-07-17-US-MTV-2/Mi8/'
    #folder = 'data/train/2021-04-22-US-SJC-1/SamsungS20Ultra/'

    #tf_gps_model.get_track_path('data/test','2020-05-28-US-MTV-2')
    #return
    #tf_gps_model.get_track_path('data/train','2020-05-14-US-MTV-1')
    #tf_gps_model.get_track_path('data/train','2020-05-14-US-MTV-2')
    #tf_gps_model.get_track_path('data/train','2021-04-22-US-SJC-1')
    #tests = glob('data/test/*')

    #tf_gps_model.get_track_path('data/test','2020-08-03-US-MTV-2')

    gt_total_score.print_gt_score()
    subset = 'train'
    tests = next(os.walk('data/'+subset))[1]
    for g in tests:
        track = g #.split('\\')[-1]
        if os.path.exists('data/'+subset+'/'+track+'/submission.csv'):
            df = pd.read_csv('data/'+subset+'/'+track+'/submission.csv')    
            if not df.isnull().values.any():
                continue
        tf_gps_model.get_track_path('data/'+subset,track)

    gt_total_score.print_gt_score()

    #make_submision.make_submission()
    #compare_track.compare_submissions("data/submission.csv", "data/submission27.csv")
    #tf_gps_model.get_track_path('data/train','2021-04-29-US-SJC-2')
    #tf_gps_model.get_track_path('data/train','2020-05-14-US-MTV-1')

def draw():
    folder = 'data/train/2020-05-14-US-MTV-2/Pixel4XLModded/'
    folder = 'data/train/2020-05-14-US-MTV-2/Pixel4/'
    folder = 'data/train/2021-04-22-US-SJC-1/Pixel4/'
    track = phone_track.PhoneTrack(folder)
    track.DrawShiftError()
    #track.DrawRelativeDistances()
    #track.DrawAccelleration()
#draw()
main()
