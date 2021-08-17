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



