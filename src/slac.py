import georinex as gr
import pickle
import os
from pathlib import Path
from glob import glob
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
import subprocess
import sys
import gzip
import shutil
    
def loadSlac(dt):
    '''
    dt = parse('2020-05-15 22:11:41')
    '''


    day  = dt.timetuple().tm_yday
    year = dt.year

    local_file = 'slac/slac'+ str(day).zfill(3)+'0.'+str(year-2000)+'o'
    if os.path.isfile(local_file):
        return local_file

    ftp = FTP('geodesy.noaa.gov')
    ftp.login()
    ftp.cwd('/cors/rinex/'+str(year)+'/'+ str(day).zfill(3) +'/slac')


    filenames = ftp.nlst()
    for f in filenames:
        if '20d' in f or '21d' in f:
            local = 'slac/'+f

            if os.path.isfile(local) == False:
                with open(local, 'wb') as local_file:
                    ftp.retrbinary('RETR ' + f, local_file.write)
            
            local_dec = local[:-3]
            print(local)
            with gzip.open(local, 'rb') as f_in:
                with open(local_dec, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            subprocess.run(['slac/rnxcmp.exe', local_dec], capture_output=True)  
        
            local = list(local_dec)      
            local[-1] = 'o'
            return "".join(local)
    
