'''
Reconstruct inflow vector from PPI
'''
import os
cd=os.getcwd()
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec
import xarray as xr
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings('ignore')

plt.close('all')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['savefig.dpi']=500

#%% Inputs
source=os.path.join(cd,'data/rt1.lidar.z02.b0.20240518.230015.user5.steer.inflow.nc')
yaw=-15#[deg] yaw misalignment
D=127#[m] rotor diameter
x_bin=-np.arange(1,12.1,2)*D#[m] bins in x
max_unc=25#[%]relative uncertainty on u,v
min_N=100#min number of points for LSQF

#graphics
N_plot=5

#%% Functions
def v_los(azi,u,v):
    return u*np.cos(np.radians(90-azi))+v*np.sin(np.radians(90-azi))

#%% Initialization

#read data
Data=xr.open_dataset(source)
time_avg=Data.time.isel(beamID=0)+(Data.time.isel(beamID=-1)-Data.time.isel(beamID=0))/2

#graphics
colormap = plt.get_cmap('coolwarm')
sel_plot = np.linspace(1, Data.scanID.values.max()-1, N_plot, dtype=int)
x_plot=(x_bin[:-1]+x_bin[1:])/2
fig=plt.figure(figsize=(18,4))
gs = GridSpec(nrows=1, ncols=N_plot+1, width_ratios=[6]*N_plot+[0.5], figure=fig)
ctr=0

#%% Main

#estimate inflow
ws = np.zeros((len(x_bin)-1,len(Data.scanID)))+np.nan
wd = np.zeros((len(x_bin)-1,len(Data.scanID)))+np.nan

for i in range(len(Data.scanID)):
    subset=Data.isel(scanID=i)
    rws=subset.wind_speed.where(subset.qc_wind_speed==0)
    azi=subset.azimuth
    i_x=0
    for x1,x2 in zip(x_bin[:-1],x_bin[1:]):
        sel=(subset.x<=x1)*(subset.x>x2)*~np.isnan(rws)

        if np.sum(sel) >= min_N:
            azi_sel=azi.where(sel).values
            rws_sel= rws.where(sel).T.values
            real=~np.isnan(rws_sel+azi_sel)
            V, cov = curve_fit(v_los, azi_sel[real], rws_sel[real])
            unc=np.sqrt(np.diag(cov))/(V[0]**2+V[1]**2)**0.5*100
            
            if np.max(unc)<max_unc:
                ws[i_x,i]=(V[0]**2+V[1]**2)**0.5
                wd[i_x,i]=(270-np.degrees(np.arctan2(V[1],V[0])))%360
            else:
                ws[i_x,i] = np.nan
                wd[i_x,i] = np.nan
        else:
            U = np.nan
            th = np.nan
        
        i_x+=1
        
    #plot velocity
    if i in sel_plot:
            
        ax = fig.add_subplot(gs[ctr])
        cf=plt.pcolor(Data.x/D,Data.y/D,-rws,cmap='coolwarm',vmin=2,vmax=7)
        norm = mcolors.Normalize(vmin=2, vmax=7) 
        plt.plot([-12,0],[np.tan(np.radians(yaw))*12,0],'--k',linewidth=2)
        
        plt.quiver(x_plot/D,x_plot*np.tan(np.radians(270-wd[:,i]))/D,
                   np.cos(np.radians(270-wd[:,i]))/D,np.sin(np.radians(270-wd[:,i]))/D,
                   facecolor='k',edgecolor='k',scale=7/D,headaxislength=4,linewidth=2,width=0.02,zorder=10)
                   
        plt.quiver(x_plot/D,x_plot*np.tan(np.radians(270-wd[:,i]))/D,
                   np.cos(np.radians(270-wd[:,i]))/D,np.sin(np.radians(270-wd[:,i]))/D,
                   facecolor=colormap(norm(ws[:,i])),edgecolor='k',cmap='coolwarm',scale=7/D,headaxislength=4,width=0.02,zorder=10)
        
        plt.xlim([-12,0.1])
        plt.ylim([-5,5])
        plt.xticks(np.arange(-10,0.2,5))
        plt.yticks(np.arange(-5,5.1,5))
        if i==sel_plot[0]:
            plt.ylabel(r'$y/D$')
        else:
            ax.set_yticklabels([])
        
        ax.set_aspect('equal')
        plt.xlabel(r'$x/D$')
        
        plt.grid()
        plt.title(f'{str(subset.time.values[0]).split("T")[1][:8]}-{str(subset.time.values[-1]).split("T")[1][:8]}',fontsize=14)
        ctr+=1
        
cbar=plt.colorbar(cf,cax=fig.add_subplot(gs[-1]),label='Radial wind speed [m s$^{-1}$]')
 
           
  