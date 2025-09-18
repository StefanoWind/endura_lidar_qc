# -*- coding: utf-8 -*-
"""
Plot sample scan
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import xarray as xr
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['savefig.dpi']=500
plt.close("all")

#%% Inputs
source_a0=os.path.join(cd,'data','rt2.lidar.z02.a0.20230815.003016.user5.nc')
source_b0=os.path.join(cd,'data','rt2.lidar.z02.b0.20230815.003016.user5.meand.2d.nc')
N=5 
D=127 #[m]

#%% Initialization
Data_a0=xr.open_dataset(source_a0)
Data_b0=xr.open_dataset(source_b0)

#%% Plots
plt.figure(figsize=(18,8))
plt.pcolor(Data_a0.time,Data_a0.range_gate,Data_a0.wind_speed.T,vmin=0,vmax=10,cmap='coolwarm')
plt.grid()
plt.xlabel('Time (UTC)')
plt.ylabel('Range gate index')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.colorbar(label='Radial wind speed [m s$^{-1}$]')

fig=plt.figure(figsize=(18,4))
gs = GridSpec(nrows=1, ncols=N+1, width_ratios=[6]*N+[0.5], figure=fig)
scans = np.linspace(Data_b0.scanID.values.min(), Data_b0.scanID.values.max(), 5, dtype=int)
for i, scan in enumerate(scans):
    subset = Data_b0.isel(scanID=scan)
    ax=fig.add_subplot(gs[i])
    cf=plt.pcolor(subset.x/D,subset.y/D,subset.wind_speed.where(subset.qc_wind_speed==0),vmin=0,vmax=10,cmap='coolwarm')
    plt.grid()
    plt.xlim([-0.2,12])
    plt.ylim([-5,5])
    ax.set_aspect('equal')
    
    plt.xlabel('$x/D$')
    if i==0:
        plt.ylabel('$y/D$')
    else:
        ax.set_yticklabels([])
    ax.set_xticks(np.arange(0,12,5))
    ax.set_yticks(np.arange(-5,5.1,5))
    plt.title(f'{str(subset.time.values[0]).split("T")[1][:8]}-{str(subset.time.values[-1]).split("T")[1][:8]}',fontsize=12)
    
cbar=plt.colorbar(cf,cax=fig.add_subplot(gs[-1]),label='Radial wind speed [m s$^{-1}$]')
