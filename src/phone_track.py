from math import isnan
from numpy.core.fromnumeric import sort
from numpy.lib.function_base import median
from tensorflow.python.ops.gen_math_ops import Range
from slac import loadSlac
from pathlib import Path
from glob import glob
import pandas as pd
import pymap3d as pm
from tqdm import tqdm
from loader import *
import numpy as np
import tensorflow as tf


import datetime
from dateutil.parser import parse
from glob import glob

from matplotlib import pyplot
from sklearn.linear_model import (
    LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor)
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import scipy
from scipy.signal import medfilt
import tensorflow_position

from coords_tools import * 
import grinex
import read_log


class PhoneTrack:
    def calculateSatScales(self):
        
        prev_dist = [NaN]*256
        prev_psev = [NaN]*256

        scales = []
        for i in range(256):
            scales.append([])

        for i in range(len(self.slac_times)):
            sats = np.array(getValuesAtTime(self.sat_poses_times, self.sat_poses, self.slac_times[i]/1000000+1000))
            
            psevdoslac = self.slac[i]

            dist_slac = np.linalg.norm(self.slac_coords-sats[:,:3], axis = -1)
            rsat = rotate_sat(sats[:,:3], dist_slac) 
            dist_slac = np.linalg.norm(self.slac_coords-rsat, axis = -1)
            rsat = rotate_sat(sats[:,:3], dist_slac) 
            dist_slac = np.linalg.norm(self.slac_coords-rsat, axis = -1)

            dist_dif = dist_slac  - prev_dist
            psev_dif = psevdoslac - prev_psev
            prev_dist = dist_slac
            prev_psev = psevdoslac

            for i in range(256):
                if np.isnan(dist_dif[i]):
                    continue
                if np.isnan(psev_dif[i]):
                    continue
                scales[i].append(dist_dif[i]/psev_dif[i])
        
        for i in range(256):
            if len(scales[i]) == 0:
                scales[i] = 1
                continue

            scales[i] = sorted(scales[i])
            scales[i] = scales[i][len(scales[i])//2]

        return scales

    def __init__(self, folder):
        files = glob(folder+"/*")
        for f in files:
            if ".pkl" in f:
                continue
            if "ground_truth.csv" in f:
                self.truepos = pd.read_csv(f)
            elif ".csv" in f:
                self.sat_poses   = myLoadSatPos(f)
            elif ".txt" in f:
                self.logs   = read_log.gnss_log_to_dataframes(f)
        
        files = glob(folder+"/supplemental/*")
        for f in files:
            if ".pkl" in f:
                continue
            if ".20o" in f or ".21o" in f :
                self.rover_times, self.rover = grinex.load_google_rinex(f, sat_registry)

        time = int(self.rover_times[0])/1000000000+timeshift
        time = datetime.datetime.utcfromtimestamp(time)

        slac_file = loadSlac(time)
        self.slac = myLoadRinexС1Сindexed(slac_file)
        self.slac_coords = np.array(myLoadRinex(slac_file).position)
        self.slac_coords_wgs = np.array(pm.ecef2geodetic(self.slac_coords[0],self.slac_coords[1],self.slac_coords[2]))

        self.slac_times = np.array([r[0] for r in self.slac])
        self.sat_poses_times = np.array([r[0] for r in self.sat_poses])

        self.slac = np.array([r[1] for r in self.slac])
        self.sat_poses = np.array([r[1] for r in self.sat_poses])

        self.scales = self.calculateSatScales()

    
    def truepos_toecef(self, row):
        lat, lon, alt = float(row['latDeg']),float(row['lngDeg']),float(row['heightAboveWgs84EllipsoidM'])-61
        return np.array(pm.geodetic2ecef(lat,lon,alt, deg = True))

    
    def DrawRelativeDistances(self):

        center =[]


        times = []
        difs = []

        raw_gps = self.logs['Raw'].to_numpy() 

        for _, row in tqdm(self.truepos.iterrows()):
            time = int(row['millisSinceGpsEpoch'])
            times.append(time)
            roverpos =   self.truepos_toecef(row)
            if len(center) == 0:
                center = roverpos
            sats = np.array(getValuesAtTime(self.sat_poses_times, self.sat_poses, time+1000))

            #psevdoslac = getValuesAtTime(self.slac_times, self.slac, time*1000000) + sats[:,3] - sats[:,5] - sats[:,6]
            #psevdorover = getValuesAtTime(self.rover_times, self.rover, time*1000000) + sats[:,3] - sats[:,4] - sats[:,5] - sats[:,6]
            #psevdoroverxls = sats[:,8] + sats[:,3] - sats[:,4] - sats[:,5] - sats[:,6]

            dist_to_sat = np.linalg.norm(center-sats[:,:3], axis = -1)
            rsat = rotate_sat(sats[:,:3], dist_to_sat) 
            dist_to_sat = np.linalg.norm(center-rsat, axis = -1)
            rsat = rotate_sat(sats[:,:3], dist_to_sat) 
            
            rover_to_sat = np.linalg.norm(roverpos-rsat, axis = -1)
            center_to_sat = np.linalg.norm(center-rsat, axis = -1)

            dif = rover_to_sat - center_to_sat
            dif = np.array(dif)
            difs.append(dif)

        difs = np.array(difs)
        times = np.array(times)

        times = (times - times[0])/1000
        lbl = []
        count = 0

        for i in range(256):
            index = ~np.isnan(difs[:,i])
            a = difs[:,i][index]
            if len(a) != 0:
                a = np.reshape(difs[:,i][index],(-1,1))
                b = np.reshape(times[index],(-1,1))
                #model = make_pipeline(PolynomialFeatures(0), RANSACRegressor())
                #model.fit(b, a)
                #difs[:,i] -= np.reshape(model.predict(np.reshape(times,(-1,1))),(-1))#+count*100
                count += 1
                #difs[:,i] = scipy.ndimage.filters.uniform_filter1d(difs[:,i], 21)
                #difs[:,i] = scipy.signal.medfilt(difs[:,i],31)
                c = 'g'
                for k,v in sat_registry.items():
                    if v == i%128:
                        lbl.append(k)
                        if k[0] == 'G':
                            c = 'b'
                        if k[0] == 'R':
                            c = 'r'
                        if k[0] == 'E':
                            c = 'k'
                        break
                #pyplot.plot( times, difs[:,i] + (i%128) *100, c)
                pyplot.plot( times, difs[:,i])
        print(count)
        pyplot.legend(lbl)
        pyplot.show()        







    def DrawAccelleration(self):
        times = []
        difs = []

        roverpos_m2 = np.full((3),NaN)
        roverpos_m1 = np.full((3),NaN)
        unac = self.logs['UncalAccel'].to_numpy() 
        unac[:,1] = ((unac[:,1] - unac[0,1])/1000000000).astype(np.int)
        unac_index = 0
        unac_bias = np.full((3),NaN)

        ungy = self.logs['UncalGyro'].to_numpy() 
        ungy[:,1] = ((ungy[:,1] - ungy[0,1])/1000000000).astype(np.int)
        ungy_bias = np.full((3),NaN)

        for _, row in tqdm(self.truepos.iterrows()):
            time = int(row['millisSinceGpsEpoch'])
            times.append(time)

            lat, lon, alt = float(row['latDeg']),float(row['lngDeg']),float(row['heightAboveWgs84EllipsoidM'])
            roverpos =   np.array(pm.geodetic2ecef(lat,lon,alt, deg = True))

            accel = roverpos_m1 -  (roverpos_m2 + roverpos)/2
            
            
            zvect = roverpos/np.linalg.norm(roverpos, axis = -1, keepdims=True)
            xvect = roverpos - roverpos_m2
            xvect = xvect/np.linalg.norm(xvect, axis = -1, keepdims=True)
            xvect = xvect - np.sum(xvect*zvect)*zvect
            xvect = xvect/np.linalg.norm(xvect, axis = -1, keepdims=True)
            yvect = np.cross(zvect, xvect)

            def trans_to_local(vect):
                return np.array([ np.sum(vect*xvect), np.sum(vect*yvect), np.sum(vect*zvect) ])

            accel = trans_to_local(accel)
            dif = []
            dif.extend(accel)


            locunac = unac[unac[:,1] == unac_index, 2:5]
            unca_coords = np.mean(locunac, axis = 0)
            if np.isnan(unac_bias[0]) :
                unac_bias = unca_coords

            dif.extend((unca_coords - unac_bias))

            locungy = ungy[ungy[:,1] == unac_index, 2:5]
            ungy_coords = np.mean(locungy, axis = 0)

            dif.extend(ungy_coords*5 + 5)
            unac_index += 1

            difs.append(dif)

            roverpos_m2 = roverpos_m1
            roverpos_m1 = roverpos

        difs = np.array(difs)
        times = np.array(times)
        
        times = (times - times[0])/1000

        lbl = [ 'X', 'Y', 'Z', 'Xa', 'Ya', 'Za', 'Xg', 'Yg', 'Zg']
        count = 0
        for i in range(len(difs[0])):
            index = ~np.isnan(difs[:,i])
            a = np.reshape(difs[:,i][index],(-1,1))
            b = np.reshape(times[index],(-1,1))
            model = make_pipeline(PolynomialFeatures(0), RANSACRegressor())
            model.fit(b, a)
            difs[:,i] -= np.reshape(model.predict(np.reshape(times,(-1,1))),(-1))#+count*100
            if(i < 6):
                difs[:,i] += i%3
            else:
                difs[:,i] += 5



            pyplot.plot( times, difs[:,i])

        print(count)
        pyplot.legend(lbl)
        pyplot.show()        


    def DrawSatDifs(self):
        times = []
        difs = []

        '''
        self.rover = myLoadRinexС1Сindexed('data/google-sdc-corrections/osr/rinex/2020-05-14-US-MTV-1.obs')
        self.rover_times = np.array([r[0] for r in self.rover])
        self.rover = np.array([r[1] for r in self.rover])
        '''
        
        basepos =  np.array(myLoadRinex('data/google-sdc-corrections/osr/rinex/2020-05-14-US-MTV-1.obs').position)

        for _, row in tqdm(self.truepos.iterrows()):
            time = int(row['millisSinceGpsEpoch'])
            times.append(time)

            lat, lon, alt = float(row['latDeg']),float(row['lngDeg']),float(row['heightAboveWgs84EllipsoidM'])
            roverpos =   np.array(pm.geodetic2ecef(lat,lon,alt, deg = True))
            #roverpos = basepos

            sats = np.array(getValuesAtTime(self.sat_poses_times, self.sat_poses, time+1000))

            psevdoslac = getValuesAtTime(self.slac_times, self.slac, time*1000000) + sats[:,3] - sats[:,5] - sats[:,6]
            psevdorover = getValuesAtTime(self.rover_times, self.rover, time*1000000)[:,0] + sats[:,3] - sats[:,4] - sats[:,5] - sats[:,6]
            psevdoroverxls = sats[:,8] + sats[:,3] - sats[:,4] - sats[:,5] - sats[:,6]

            dist_to_sat = np.linalg.norm(self.slac_coords-sats[:,:3], axis = -1)
            rsat = rotate_sat(sats[:,:3], dist_to_sat) 
            dist_to_sat = np.linalg.norm(self.slac_coords-rsat, axis = -1)
            rsat = rotate_sat(sats[:,:3], dist_to_sat) 

            dif = []
            
            dist_to_sat = np.linalg.norm(self.slac_coords-rsat, axis = -1)
            ds = dist_to_sat - psevdoslac
            
            dist_to_sat = np.linalg.norm(roverpos-rsat, axis = -1)
            dr = dist_to_sat - psevdorover
            
            '''
            index = ~np.isnan(d)
            a = d[index]
            a = sorted(a)
            d -= a[len(a)//2]
            '''

            dif.extend(dr-ds)
            #dif.extend(ds[:128])
            #dif.extend(dr[:128])
            dif = np.array(dif)
            difs.append(dif)

        times = times[5:-5]
        difs = difs[5:-5]
        difs = np.array(difs)
        times = np.array(times)
        difs[np.abs(difs) > 100000000] = NaN

        times = (times - times[0])/(times[-1]- times[0]) - 0.5
        lbl = []
        count = 0
        for i in range(256):
            index = ~np.isnan(difs[:,i])
            a = difs[:,i][index]
            if len(a) != 0:
                a = np.reshape(difs[:,i][index],(-1,1))
                b = np.reshape(times[index],(-1,1))
                #model = make_pipeline(PolynomialFeatures(0), RANSACRegressor())
                #model.fit(b, a)
                #difs[:,i] -= np.reshape(model.predict(np.reshape(times,(-1,1))),(-1))#+count*100
                count += 1
                #difs[:,i] = scipy.ndimage.filters.uniform_filter1d(difs[:,i], 21)
                #difs[:,i] = scipy.signal.medfilt(difs[:,i],31)
                c = 'g'
                for k,v in sat_registry.items():
                    if v == i%128:
                        lbl.append(k)
                        if k[0] == 'G':
                            c = 'b'
                        if k[0] == 'R':
                            c = 'r'
                        if k[0] == 'E':
                            c = 'k'
                        break
                #pyplot.plot( times, difs[:,i] + (i%128) *100, c)
                pyplot.plot( times, difs[:,i])
        print(count)
        pyplot.legend(lbl)
        pyplot.show()        

    def DrawSatError(self):
        times = []
       
        difs = []
        NaN = float("NaN")
        sat_distances_accum = []
        for i in range(256):
            sat_distances_accum.append([])

        sat_name_map = {}
        for k,v in sat_registry.items():
            sat_name_map[v] = k

        for _, row in tqdm(self.truepos.iterrows()):
            time = int(row['millisSinceGpsEpoch'])

            lat, lon, alt = float(row['latDeg']),float(row['lngDeg']),float(row['heightAboveWgs84EllipsoidM'])
            roverpos =   np.array(pm.geodetic2ecef(lat,lon,alt, deg = True))

            sats = np.array(getValuesAtTime(self.sat_poses_times, self.sat_poses, time+1000))

            psevdoslac = getValuesAtTime(self.slac_times, self.slac, time*1000000) 
            psevdorover = getValuesAtTime(self.rover_times, self.rover, time*1000000)
            
            dist_slac_sat = np.linalg.norm(self.slac_coords-sats[:,:3], axis = -1)
            rsat = rotate_sat(sats[:,:3], dist_slac_sat) 
            dist_slac_sat = np.linalg.norm(self.slac_coords-rsat, axis = -1)
            rsat = rotate_sat(sats[:,:3], dist_slac_sat) 

            dist_slac_sat  = np.linalg.norm(self.slac_coords-rsat, axis = -1)
            dist_rover_sat = np.linalg.norm(roverpos-rsat, axis = -1)

            dist_slac_err = dist_slac_sat - psevdoslac
            #psevdorover += dist_slac_err
            dist_rover_err = psevdorover - dist_rover_sat

            dif =  dist_rover_err
            for i in range(256):
                if i not in sat_name_map or sat_name_map[i][0] != 'G':
                    dif[i] = NaN

                if ~np.isnan(dif[i]):
                    sat_distances_accum[i].append(dif[i])
                else:
                    if len(sat_distances_accum[i]) > 0:
                        ac = sorted(sat_distances_accum[i])
                        sat_distances_accum[i] = []
                        med = ac[len(ac)//2]
                        for k in range(len(ac)):
                            difs[-1-k][i] -= med

            difs.append(dif)
            times.append(time)

        for i in range(256):
            if len(sat_distances_accum[i]) > 0:
                ac = sorted(sat_distances_accum[i])
                sat_distances_accum[i] = []
                med = ac[len(ac)//2]
                for k in range(len(ac)):
                    difs[-1-k][i] -= med
        '''
        for dif in difs:
            index = ~np.isnan(dif)
            a = dif[index]
            if len(a) != 0:
                a = sorted(a)
                dif -= a[len(a)//2]
                a -= a[len(a)//2]
        '''

        #real_dist = np.array(real_dist)
        #difs = difs[:-200]
        #times = times[:-200]

        difs = np.array(difs)
        difs_1d = np.reshape(difs,(-1))
        index = ~np.isnan(difs_1d)
        difs_1d = difs_1d[index]
        print(np.mean(np.abs(difs_1d)))
        #errors_dist = real_dist-pred_dist
        times = np.array(times)
        times = (times - times[0]) / 1000
        #real_dist = real_dist - real_dist[0]
        #pred_dist = pred_dist - pred_dist[0]

        
        lbl = []
        count = 0
        error_list = []
        for i in range(256):
            index = ~np.isnan(difs[:,i])
            a = difs[:,i][index]
            #b = pred_dist[:,i][index]
            if len(a) > 10:
                '''
                a = np.reshape(a,(-1,1))
                b = np.reshape(b,(-1,1))
                try:
                    model = make_pipeline(PolynomialFeatures(1), RANSACRegressor())
                    #model.fit(b, a)
                    #real_dist[:,i] -= np.reshape(model.predict(np.reshape(pred_dist[:,i],(-1,1))),(-1))#+count*100
                except:
                    continue
                '''
                count += 1
                key = ''
                for k,v in sat_registry.items():
                    if v == i:
                        key = k
                #if key[0] == 'R':
                #    continue
                #difs[:,i] = scipy.ndimage.filters.uniform_filter1d(difs[:,i], 21)
                #difs[:,i] = scipy.signal.medfilt(difs[:,i],31)
                pyplot.plot( times, difs[:,i])
                lbl.append(key)
                error_list.extend(a)
        #pyplot.ylim((-20, 20))
        bins = 1000
        hist, bin_edges = np.histogram(error_list, bins=bins)
        print(np.mean(np.abs(error_list)))
        for i in range(0,1000):
            if abs(bin_edges[i]) < 0.5:
                print(i, bin_edges[i], hist[i], hist[i]/len(error_list))
        pyplot.legend(lbl)

        pyplot.show()        
    
    
    def DrawShiftError(self):
        times = []
        difs = []
        NaN = float("NaN")

        sat_name_map = {}
        for k,v in sat_registry.items():
            sat_name_map[v] = k

        prev_rover_pos = np.array([NaN,NaN,NaN])
        prev_sat_pos = np.array([[NaN,NaN,NaN]]*256)
        prev_psevdorover = np.array([NaN]*256)
        shift = [0,0,0,0]
        
        times_truepos = []
        truepos = []
        counts = [0,0,0,0,0,0,0,0,0,0,0,0]
        last_ok = np.zeros((256))
        for _, row in tqdm(self.truepos.iterrows()):
            time = int(row['millisSinceGpsEpoch'])
            lat, lon, alt = float(row['latDeg']),float(row['lngDeg']),float(row['heightAboveWgs84EllipsoidM'])
            roverpos =   np.array(pm.geodetic2ecef(lat,lon,alt-61, deg = True))
            truepos.append(roverpos)
            times_truepos.append(time*1000000)

        for j in tqdm(range(len(self.rover_times))):

            time = self.rover_times[j]
            psevdorover = self.rover[j]

            psevdorover_sgn  = psevdorover[:,1]
            psevdorover = psevdorover[:,0] * self.scales
            sats = np.array(getValuesAtTime(self.sat_poses_times, self.sat_poses, time/1000000+1000))
            roverpos = np.array(getValuesAtTime(times_truepos, truepos, time))


            dist_truepos = np.linalg.norm(roverpos-sats[:,:3], axis = -1)
            rsat = rotate_sat(sats[:,:3], dist_truepos) 
            dist_truepos = np.linalg.norm(roverpos-rsat, axis = -1)
            rsat = rotate_sat(sats[:,:3], dist_truepos) 

            dist_rover_sat = np.linalg.norm(roverpos-rsat, axis = -1)
            dist_rover_prev_sat = np.linalg.norm(prev_rover_pos-prev_sat_pos, axis = -1)
            dist_rover_this_sat = np.linalg.norm(prev_rover_pos-rsat, axis = -1)

            pred_change = dist_rover_this_sat - dist_rover_prev_sat
            real_change = psevdorover - prev_psevdorover
            distances_changes = - (real_change - pred_change)
            distances_changes = np.reshape(distances_changes,(256,1))

            sat_vect = (rsat-roverpos)/np.linalg.norm(rsat - roverpos, axis = -1, keepdims=True)

            #distances_changes = np.sum(sat_vect*(roverpos-prev_rover_pos),axis=1,keepdims=True)


            if np.isnan(shift[0]):
                shift = [0,0,0,0]

            zvect = roverpos/np.linalg.norm(roverpos, axis = -1, keepdims=True)
            xvect = roverpos - prev_rover_pos
            l = np.sqrt(np.sum(xvect*xvect))
            if l < 0.1:
                xvect = np.array([0.,0.,1.])
            else:
                xvect /= l
            xvect -= np.sum(xvect*zvect, keepdims=True)*zvect
            xvect /= np.sqrt(np.sum(xvect*xvect))
            yvect = np.cross(zvect, xvect)

            def trans_to_local(vect):
                return np.array([ np.sum(vect*xvect), np.sum(vect*yvect), np.sum(vect*zvect) ])
            
            sat_vect_local = []
            for i in range(256):
                sat_vect_local.append(trans_to_local(sat_vect[i]))
            
            sat_vect_local = np.array(sat_vect_local)
            if True:
                shift, err = calc_shift_fromsat2d(sat_vect_local, distances_changes, shift)
                if len(err) == 0 or np.sum(err < 0.1) < 8:
                    shift, err = calc_shift_fromsat(sat_vect_local, distances_changes, shift)
                    shift = np.array([NaN,NaN,NaN,NaN])
                else:
                   shift = np.reshape(shift,(-1))

                localtrans = trans_to_local(roverpos - prev_rover_pos)
                dif1 = localtrans - shift[:3]
                if np.linalg.norm(dif1) > 1:
                    sat_dir_copy = sat_vect_local.copy()
                    cur_shifts = np.sum(sat_dir_copy*np.reshape(localtrans,(1,3)), axis=1)
                    rows = cur_shifts - np.reshape(distances_changes,(-1))
                    ind = ~np.isnan(rows)
                    med = np.median(rows[ind])
                    rows[ind] = np.abs(rows[ind] - med)
                    combined = np.vstack((rows, err)).T
                    shift, err = calc_shift_fromsat2d(sat_vect_local, distances_changes, shift)

                sat_err = np.abs(np.sum(sat_vect_local*shift[:3], axis = -1) - np.reshape(distances_changes,(-1)) + shift[3])
                for i in range(256):
                    if (~np.isnan(sat_err[i]) and sat_err[i] > 0.3) or (np.isnan(prev_psevdorover[i]) and ~np.isnan(psevdorover[i])):
                        if last_ok[i] == 1:
                            counts[1] += 1
                        last_ok[i] = 0
                    elif (~np.isnan(sat_err[i]) and sat_err[i] <= 0.3):
                        counts[0] += 1
                        last_ok[i] = 1


            else:
                shift, err = calc_shift_fromsat(sat_vect, distances_changes, shift)
                localtrans = roverpos - prev_rover_pos
                dif1 = localtrans - shift[:3]
                if np.linalg.norm(dif1) > 0.2:
                    shift, err = calc_shift_fromsat(sat_vect, distances_changes, shift)


            real_shift2d = np.reshape(localtrans,(1,3))
            scalar_mult = real_shift2d*sat_vect_local
            dif = np.sum(scalar_mult,axis = -1) + np.reshape(distances_changes,(-1))
            index = ~np.isnan(dif)
            difnn = sorted(dif[index])
            time_shift = 0
            if len(difnn) != 0:
                time_shift = difnn[len(difnn)//2]
            real_shift = [real_shift2d[0,0],real_shift2d[0,1],real_shift2d[0,2],time_shift]
            distances_changes = np.reshape(distances_changes,(-1))
            index = ~np.isnan(distances_changes)
            if len(sat_vect_local[index]) > 0 and len(sat_vect_local[index]) > 4:
                checker = pr_shift(sat_vect_local[index], distances_changes[index],100)
                err2 = checker(real_shift)
            else:
                err2 = None
            
            dif2 = []
            err = np.sort(np.reshape(err,(-1)))
            #if len(err) > 5:
            #    e = np.mean(err[:6]) * 10
            #    if (e > 0.1):
            #        dif1 = [NaN, NaN, NaN]
                #dif2.append(e)
            #else:
                #dif2.append(NaN)
            dif2.extend(dif1)
            if abs(time_shift - shift[3]) > 5:
                time_shift += 0.01
            #dif2.append(time_shift)
            dif2.append(shift[2])
            #dif2.append(shift[2])
            #dif2 = dif2[:2]
            if np.sum(np.abs(dif2)) > 0.2:
                prev_sat_pos = rsat
            difs.append(dif2)
            times.append(time)

            '''
            #difs.append(dif2)
            dif = np.reshape(distances_changes,(-1))
            index = ~np.isnan(dif)
            difnn = sorted(dif[index])
            time_shift = 0
            if len(difnn) != 0:
                time_shift = difnn[len(difnn)//2]

            distances_changes -= time_shift

            asbchanges = np.abs(distances_changes[index])
            asbchanges = np.sort(asbchanges, axis = None)
            numgood = 0
            while numgood < len(difnn) and asbchanges[numgood] < 0.3:
                numgood += 1

            if numgood > 3 and numgood < 6:
                dif = np.reshape(distances_changes,(-1))
                difnn = sorted(dif[index])

            if numgood >= 10:
                counts[10] += 1
            else:
                counts[numgood] += 1

            #time_shift = 0
            difs.append(distances_changes)
            times.append(time)
            '''

            prev_sat_pos = rsat
            prev_rover_pos = roverpos
            prev_psevdorover = psevdorover

        print(counts)
        difs = np.array(difs)
        times = np.array(times)
        times = (times - times[0]) / 1000
        
        lbl = []
        count = 0
        error_list = []
        for i in range(len(difs[0])):
            index = ~np.isnan(difs[:,i])
            a = difs[:,i][index]
            #b = pred_dist[:,i][index]
            if len(a) > 10:
                '''
                a = np.reshape(a,(-1,1))
                b = np.reshape(b,(-1,1))
                try:
                    model = make_pipeline(PolynomialFeatures(1), RANSACRegressor())
                    #model.fit(b, a)
                    #real_dist[:,i] -= np.reshape(model.predict(np.reshape(pred_dist[:,i],(-1,1))),(-1))#+count*100
                except:
                    continue
                '''
                count += 1
                key = ''
                for k,v in sat_registry.items():
                    if v == i:
                        key = k
                #if key[0] == 'R':
                #    continue
                #difs[:,i] = scipy.ndimage.filters.uniform_filter1d(difs[:,i], 21)
                #difs[:,i] = scipy.signal.medfilt(difs[:,i],31)
                pyplot.plot( times, difs[:,i])
                lbl.append(key)
                error_list.extend(a)
        #pyplot.ylim((-20, 20))
        bins = 30
        error_list = np.array(error_list)
        error_list[error_list>1] = 1
        error_list[error_list<-1] = -1
        hist, bin_edges = np.histogram(error_list, bins=bins)
        print(np.mean(np.abs(error_list)))
        print(len(a)/len(times))
        for i in range(0,30):
            if abs(bin_edges[i]) < 0.5:
                print(i, bin_edges[i], hist[i], hist[i]/len(error_list))
        pyplot.legend(lbl)

        pyplot.show()        
    
    def DrawPosErrorTensorflow(self):

        def calculate_error(positions):
            total_error = 0
            total_samples = 0
            err_list = []
            numepoch = 1
            times = []
            difs = []

            positions += true_positions[1]-positions[1]
            for i in range(15):
                roverpos =   np.array(pm.ecef2geodetic(true_positions[i,0],true_positions[i,1],true_positions[i,2]))
                roverpos_wgs =   np.array(pm.ecef2geodetic(positions[i,0],positions[i,1],positions[i,2]))
                print(roverpos, roverpos_wgs)
            
            for _, row in tqdm(self.truepos.iterrows()):
                time = int(row['millisSinceGpsEpoch'])
                times.append(time)

                lat, lon, alt = float(row['latDeg']),float(row['lngDeg']),float(row['heightAboveWgs84EllipsoidM'])
                roverpos =   np.array(pm.geodetic2ecef(lat,lon,alt))
                roverpos_wgs =   np.array(pm.ecef2geodetic(positions[numepoch,0],positions[numepoch,1],positions[numepoch,2]))
                dif = positions[numepoch,:]  - roverpos
                er = calc_haversine(roverpos_wgs[0],roverpos_wgs[1],lat,lon)
                if ~np.isnan(er):
                    total_samples += 1
                    total_error += er
                    err_list.append(er)
                    if er > 100:
                        er = 0
                '''
                dif = []
                dif.extend(2*initial_positions[numepoch]-initial_positions[numepoch-1]-initial_positions[numepoch+1])
                dif.extend(2*positions[numepoch]-positions[numepoch-1]-positions[numepoch+1])
                dif = dif[0::3]
                '''
                difs.append(dif)
            
                numepoch += 1

            err_list = sorted(err_list)
            print("Mean squared error ", total_error/total_samples)
            print("90 error ", err_list[total_samples*9//10])
            print("50 error ", err_list[total_samples*5//10])
            print("error ", (err_list[total_samples*5//10]+err_list[total_samples*9//10])/2)

            times = times[5:-5]
            difs = difs[5:-5]
            difs = np.array(difs)
            times = np.array(times)
            difs[np.abs(difs) > 100000000] = NaN

            times = (times - times[0])/(times[-1]- times[0]) - 0.5
            lbl = []
            count = 0
            for i in range(len(difs[0])):
                index = ~np.isnan(difs[:,i])
                a = difs[:,i][index]
                if len(a) != 0:
                    a = np.reshape(difs[:,i][index],(-1,1))
                    b = np.reshape(times[index],(-1,1))

                    #model1 = make_pipeline(PolynomialFeatures(2), RANSACRegressor())
                    #model1.fit(b, a)
                    #difs[:,i] -= np.reshape(model1.predict(np.reshape(times,(-1,1))),(-1))#+count*100
                    count += 1
                    #difs[:,i] = scipy.ndimage.filters.uniform_filter1d(difs[:,i], 21)
                    #difs[:,i] = scipy.signal.medfilt(difs[:,i],31)
                    pyplot.plot( times, difs[:,i])
                    lbl.append('G'+str(i).zfill(2))
            #pyplot.ylim((-1000, 1000))
            pyplot.legend(lbl)

            pyplot.show()        
            return difs,times
        def draw_sat_error(model, totalepoch):

            datapred = tf.data.Dataset.from_tensor_slices(((epochs,sat_ids, seg_ids, sat_pos),distances)).batch(1024*32)
            distances_pred = model.predict(datapred)
            difs = np.zeros((totalepoch,256))
            difs[:,:] = NaN
            times = np.arange(start = 0, stop = totalepoch)
            difs[:,40] = model.get_layer("time_bias").get_weights()[0][:,0]

            for i in range(len(distances_pred)):
                difs[epochs[i],sat_ids[i]] = distances[i] - distances_pred[i][0]

            
            lbl = []
            count = 0
            for i in range(len(difs[0])):
                index = ~np.isnan(difs[:,i])
                a = difs[:,i][index]
                if len(a) != 0:
                    pyplot.plot( times, difs[:,i])
                    lbl.append('G'+str(i).zfill(2))
            
            pyplot.legend(lbl)
            pyplot.show()        
            return difs,times

        pos = [0,0,0,0]

        numepoch = 1

        epochs = []
        seg_ids = []
        sat_ids = []
        sat_pos = []
        distances = []
        sat_umbiq = [-1]*256
        cur_umbiq = 0
        umbiq_init = []

        initial_positions = []
        true_positions = []

        pos = [0,0,0,0]
        initial_positions.append([0,0,0])
        true_positions.append([0,0,0])

        sat_name_map = {}
        for k,v in sat_registry.items():
            sat_name_map[v] = k

        scales = self.scales

        for _, row in tqdm(self.truepos.iterrows()):

            time = int(row['millisSinceGpsEpoch'])

            sats = np.array(getValuesAtTime(self.sat_poses_times, self.sat_poses, time+1000))
            
            psevdoslac = getValuesAtTime(self.slac_times, self.slac, time*1000000)*scales
            psevdorover = getValuesAtTime(self.rover_times, self.rover, time*1000000)*scales
            psevdoroverxls = sats[:,8] + sats[:,3] - sats[:,4] - sats[:,5] - sats[:,6]

            dist_slac = np.linalg.norm(self.slac_coords-sats[:,:3], axis = -1)
            rsat = rotate_sat(sats[:,:3], dist_slac) 
            dist_slac = np.linalg.norm(self.slac_coords-rsat, axis = -1)
            rsat = rotate_sat(sats[:,:3], dist_slac) 

            if np.isnan(pos[0]):
                pos = [0,0,0,0]
            
            pos,err = calc_pos_fix(rsat,psevdoroverxls, np.ones((256)), pos)
            lat, lon, alt = float(row['latDeg']),float(row['lngDeg']),float(row['heightAboveWgs84EllipsoidM'])
            roverpos =   np.array(pm.geodetic2ecef(lat,lon,alt))
            true_positions.append(roverpos)            
            
            roverpos = pos[:3]
            dist_slac = np.linalg.norm(self.slac_coords-rsat, axis = -1)
            dist_rover = np.linalg.norm(roverpos-rsat, axis = -1)

            dist1err = dist_slac - psevdoslac
            psevdorover += dist1err
            
            initial_positions.append(roverpos)            

            for i in range(256):
                if np.isnan(psevdorover[i]): # or i not in sat_name_map or sat_name_map[i][0] != 'G':
                    sat_umbiq[i] = -1
                    continue

                if sat_umbiq[i] == -1:
                    sat_umbiq[i] = cur_umbiq
                    umbiq_init.append(psevdorover[i]-dist_rover[i])
                    cur_umbiq += 1
                
                epochs.append(numepoch)
                sat_ids.append(i)
                seg_ids.append(sat_umbiq[i])
                sat_pos.append(rsat[i])
                distances.append(psevdorover[i])

            numepoch += 1

        initial_positions.append([0,0,0])
        true_positions.append([0,0,0])
        initial_positions[0] = initial_positions[1]
        initial_positions[-1] = initial_positions[-2]
        initial_positions = np.array(initial_positions)

        model, poses, accel = tensorflow_position.createModel(numepoch+1, 256, cur_umbiq, initial_positions, umbiq_init)
        sat_pos = np.array(sat_pos).astype(np.float64)
        distances = np.array(distances).astype(np.float64)
        data = tf.data.Dataset.from_tensor_slices(((epochs,sat_ids, seg_ids, sat_pos),distances)).repeat().shuffle(1024*32).batch(1024*32)
        true_positions = np.array(true_positions)
        
        model.summary()

        model.get_layer("time_bias").trainable = False
        model.get_layer("positions").trainable = False

        model.compile(optimizer=tf.keras.optimizers.Adam(1000),loss = 'MSE')
        model.fit(data, epochs=1, steps_per_epoch=128, callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', verbose=1, patience = 20, min_delta=0.01)])
        #draw_sat_error(model,numepoch+1)
        model.get_layer("time_bias").trainable = True
        model.compile(optimizer=tf.keras.optimizers.Adam(100),loss = 'MSE')
        model.fit(data, epochs=1, steps_per_epoch=128, callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', verbose=1, patience = 20, min_delta=0.01)])
        model.compile(optimizer=tf.keras.optimizers.Adam(10),loss = 'MSE')
        model.fit(data, epochs=1, steps_per_epoch=128, callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', verbose=1, patience = 20, min_delta=0.01)])
        #draw_sat_error(model,numepoch+1)

        model.compile(optimizer=tf.keras.optimizers.Adam(1),loss = 'MSE')
        model.fit(data, epochs=1, steps_per_epoch=512, callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', verbose=1, patience = 20, min_delta=0.01)])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.1),loss = 'MSE')
        model.fit(data, epochs=1, steps_per_epoch=512, callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', verbose=1, patience = 20, min_delta=0.01)])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.01),loss = 'MSE')
        model.fit(data, epochs=1, steps_per_epoch=512, callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', verbose=1, patience = 20, min_delta=0.01)])
        #draw_sat_error(model,numepoch+1)
        #model.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss = 'MAE')
        #model.fit(data, epochs=1, steps_per_epoch=512, callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', verbose=1, patience = 20, min_delta=0.01)])
        #model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),loss = 'MAE')
        #model.fit(data, epochs=1, steps_per_epoch=512, callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', verbose=1, patience = 20, min_delta=0.01)])
        

        positions = poses(np.arange(numepoch+1)).numpy()
        print(self.slac_coords_wgs)
        #calculate_error(positions)
        model.get_layer("positions").trainable = True
        tensorflow_position.set_position_regulizer(model)

        #model.compile(optimizer=tf.keras.optimizers.Adam(10),loss = 'MAE')
        #model.fit(data, epochs=1, steps_per_epoch=128, callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', verbose=1, patience = 20, min_delta=0.001)])
        #model.compile(optimizer=tf.keras.optimizers.Adam(1),loss = 'MAE')
        #model.fit(data, epochs=1, steps_per_epoch=256, callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', verbose=1, patience = 20, min_delta=0.001)])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.1),loss = 'MAE')
        model.fit(data, epochs=1, steps_per_epoch=256, callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', verbose=1, patience = 20, min_delta=0.001)])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.01),loss = 'MAE')
        model.fit(data, epochs=1, steps_per_epoch=256, callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', verbose=1, patience = 20, min_delta=0.001)])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss = 'MAE')
        model.fit(data, epochs=1, steps_per_epoch=256, callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', verbose=1, patience = 20, min_delta=0.001)])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),loss = 'MAE')
        model.fit(data, epochs=1, steps_per_epoch=256, callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', verbose=1, patience = 20, min_delta=0.001)])

        positions = poses(np.arange(numepoch+1)).numpy()
        calculate_error(positions)
        draw_sat_error(model,numepoch+1)


        model.compile(optimizer=tf.keras.optimizers.Adam(0.1),loss = 'MAE')
        model.fit(data, epochs=256, callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', verbose=1)])
