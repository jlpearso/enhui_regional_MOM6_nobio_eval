# Packages -----------------------------------------------#

# Data Analysis
import os
import glob
import xarray as xr
# import rioxarray as rxr
import numpy as np
import pandas as pd
import matplotlib.dates as dates
from matplotlib.font_manager import FontProperties
import matplotlib as mpl
import os
import sys
import datetime as dt
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from scipy import stats
import geopy.distance as gpds
from matplotlib.colors import LogNorm
import itertools
import gsw as gsw
import scipy.interpolate as sp

# Plotting
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter
from matplotlib.lines import Line2D
import cmocean
import harmonica as hm
from IPython.display import HTML
from matplotlib.animation import FuncAnimation


# Timing Processes
import time
from tqdm import tqdm


print('Default libraries loaded.')
