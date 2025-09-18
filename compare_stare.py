# -*- coding: utf-8 -*-
"""
Compare stare data
"""
import os
cd=os.path.dirname(__file__)
import sys
import numpy as np
import xarray as xr
import matplotlib
from matplotlib import pyplot as plt
import glob
from matplotlib.gridspec import GridSpec
import warnings
import yaml
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['savefig.dpi']=500
plt.close("all")

#%% Inputs
#users inputs
if len(sys.argv)==1:
    path_config=os.path.join(cd,'configs','config.yaml')
    time_step=1#[s] time step
    max_time_diff=2#[s] maximum time gap
else:
    path_config=sys.argv[1]
    time_step=int(sys.argv[2])
    max_time_diff=int(sys.argv[3])

range_sel=np.array([100,500,1000])#[m]

#%% Functions
def plot_lin_fit(x, y, bins=50, cmap='Greys',ax=None,cax=None,legend=True,limits=None):
    from scipy.stats import linregress
    
    # Remove NaNs
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]

    # Linear regression
    slope, intercept, r_value, _, _ = linregress(x, y)
    y_fit = slope * x + intercept
    bias=np.nanmean(y-x)
    rmsd = np.sqrt(np.mean((y - y_fit)**2))
    r_squared = r_value**2

    # Plot setup
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # 2D histogram
    if limits is None:
        h = ax.hist2d(x, y, bins=bins, cmap=cmap)
    else:
        h = ax.hist2d(x, y, bins=bins, cmap=cmap,vmin=limits[0],vmax=limits[1])
    if cax is not None:
        plt.colorbar(h[3], ax=ax,cax=cax, label='Counts')

    # Regression line
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot([np.min(x),np.max(x)],[np.min(x),np.max(x)],'--b',label='1:1')
    ax.plot(x_line, slope * x_line + intercept, color='red', linewidth=2, label='Linear fit')
    
    # Stats textbox
    textstr = '\n'.join((
        f'Intercept: {intercept:.2f}',
        f'Slope: {slope:.2f}',
        r'$R^2$: {:.2f}'.format(r_squared),
        f'Bias: {bias:.2f}',
        f'RMS: {rmsd:.2f}',
    ))
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_aspect("equal")
    if legend:
        plt.legend(draggable=True)

#%% Initialization
#configs
with open(path_config, 'r') as fid:
    config = yaml.safe_load(fid)        

rws_all={}

os.makedirs(os.path.join(cd,'figures'),exist_ok=True)

#%% Main
for s in config['lidars']:
    files=sorted(glob.glob(config['sources'][s]))
    rws_all[s]=xr.DataArray()
    for f in files:
        Data=xr.open_dataset(f)
        
        #asign time as coordinate
        Data=Data.assign_coords(scanID=Data.time)
        Data=Data.drop_vars(["time", "beamID"]).squeeze(drop=True).rename({"scanID": "time"})
        
        #define uniform time distribution
        t1 = np.datetime64(str(Data.time.min().values)[:14]+'00:00')
        t2 = t1+np.timedelta64(3600,'s')
        time = np.arange(t1, t2, np.timedelta64(time_step, "s"))
        tnum=(time-np.datetime64('1970-01-01T00:00:00'))/np.timedelta64(1,'s')
        
        #calculate the difference between the interpolation time and the nearest available time
        Data['tnum']=(Data.time-np.datetime64('1970-01-01T00:00:00'))/np.timedelta64(1,'s')
        Data['tnum']=Data['tnum'].expand_dims({"range":Data.range})
        Data['tnum']=Data['tnum'].where(Data.qc_wind_speed==0)
        time_diff=Data['tnum'].interp(time=time,method="nearest")-tnum
        
        #interpolate in time and range
        rws_int = Data.wind_speed.where(Data.qc_wind_speed==0).interp(time=time).where(np.abs(time_diff)<max_time_diff).interp(range=range_sel,method='nearest')
        
        if 'time' not in rws_all[s].coords:
            rws_all[s]=rws_int
        else:
            rws_all[s]=xr.concat([rws_all[s],rws_int],dim='time')
            
        print(f'{f} done',flush=True)
        
#%% Plots
plt.close('all')

for r in range_sel:
    bar_done=False
    fig=plt.figure(figsize=(16,10))
    gs = GridSpec(nrows=len(config['lidars'])-1, ncols=len(config['lidars']), width_ratios=[6]*(len(config['lidars'])-1)+[0.5], figure=fig)
    i1=0
    for s1 in config['lidars']:
        i2=i1
        for s2 in config['lidars'][i1+1:]:
            
            rws1,rws2=xr.align(rws_all[s1],rws_all[s2])

            ax=fig.add_subplot(gs[i1,i2])
            if np.sum(~np.isnan(rws1.sel(range=r)+rws1.sel(range=r)))>0:
                if bar_done==False:
                    cax=fig.add_subplot(gs[:,-1])
                else:
                    cax=None
                    bar_done=True
                plot_lin_fit(rws1.sel(range=r).values, rws2.sel(range=r).values,ax=ax,cax=cax,legend=(i1==0)*(i2==0),limits=[0,500])
            else:
                i2+=1
                continue
            ax.set_xlabel(f'RWS (Lidar {s1}) '+r'[m s$^{-1}$]')
            ax.set_ylabel(f'RWS (Lidar {s2}) '+r'[m s$^{-1}$]')
            ax.set_xlim([-2,2])
            ax.set_ylim([-2,2])
            ax.grid(True)
            
            i2+=1
        i1+=1
    
    plt.tight_layout()
    plt.savefig(os.path.join(cd,'figures',f'{r}_stares_linfit.png'))
    plt.close()
        
    
