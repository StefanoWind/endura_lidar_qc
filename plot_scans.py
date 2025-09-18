# -*- coding: utf-8 -*-
"""
Plot scan geometry
"""

import os
cd=os.getcwd()
import numpy as np
from matplotlib import pyplot as plt
import xarray as xr
import glob as glob
import warnings
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['savefig.dpi'] = 500

warnings.filterwarnings('ignore')
plt.close('all')

#%% Inputs
source='C:/Users/sletizia/OneDrive - NREL/Desktop/Main/ENDURA/awaken_lidar_processing/data/awaken/rt1.lidar.z02.b0'
scans=['inflow.turb','meand.3d','inflow.stats','wake.stats','bloc','rhi','meand.2d','steer.inflow','steer.wake.2d','steer.wake.3d']
# scans=['farm.wake']

#graphics
D=127#[m] turbine diameter
H=90#[m] hub height
rmax=D*5#[m] max dist

#%% Functions
    
def draw_turbine_3d(ax,x,y,z,D,H,yaw):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # Correct import
    from stl import mesh  # Correct import for Mesh
    
    # Load the STL file of the 3D turbine model
    turbine_mesh = mesh.Mesh.from_file(os.path.join(cd,'blades.stl'))
    tower_mesh = mesh.Mesh.from_file(os.path.join(cd,'tower.stl'))
    nacelle_mesh = mesh.Mesh.from_file(os.path.join(cd,'nacelle.stl'))

    #translate
    translation_vector = np.array([-125, -110, -40])
    turbine_mesh.vectors += translation_vector

    translation_vector = np.array([-125, -95, -150])
    tower_mesh.vectors += translation_vector

    translation_vector = np.array([-125, -100,-10])
    nacelle_mesh.vectors += translation_vector

    #rescale
    scaling_factor = 1/175*D
    turbine_mesh.vectors *= scaling_factor

    scaling_factor = 1/250*D
    scaling_factor_z=1/0.6*H/D
    tower_mesh.vectors *= scaling_factor
    tower_mesh.vectors[:, :, 2] *= scaling_factor_z

    scaling_factor = 1/175*D
    nacelle_mesh.vectors *= scaling_factor

    #rotate
    theta = np.radians(180+yaw)  
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,             0,             1]
    ])

    turbine_mesh.vectors = np.dot(turbine_mesh.vectors, rotation_matrix)
    tower_mesh.vectors = np.dot(tower_mesh.vectors, rotation_matrix)
    nacelle_mesh.vectors = np.dot(nacelle_mesh.vectors, rotation_matrix)

    #translate
    translation_vector = np.array([x, y, z])
    turbine_mesh.vectors += translation_vector
    tower_mesh.vectors += translation_vector
    nacelle_mesh.vectors += translation_vector


    # Extract the vertices from the rotated STL mesh
    faces = turbine_mesh.vectors  # Each face of the mesh

    # Create a Poly3DCollection for the faces
    poly_collection = Poly3DCollection(faces, facecolors=(0,0,0,0.8), linewidths=1, edgecolors=None, alpha=0.5)
    ax.add_collection3d(poly_collection)

    # Get the scale from the mesh to fit it properly
    scale = np.concatenate([turbine_mesh.points.min(axis=0), turbine_mesh.points.max(axis=0)])

    # Extract the vertices from the rotated STL mesh
    faces = tower_mesh.vectors  # Each face of the mesh

    # Create a Poly3DCollection for the faces
    poly_collection = Poly3DCollection(faces, facecolors=(0,0,0,0.8), linewidths=1, edgecolors=None, alpha=0.5)
    ax.add_collection3d(poly_collection)

    # Extract the vertices from the rotated STL mesh
    faces = nacelle_mesh.vectors  # Each face of the mesh

    # Create a Poly3DCollection for the faces
    poly_collection = Poly3DCollection(faces, facecolors=(0,0,0,0.8), linewidths=1, edgecolors=None, alpha=0.5)
    ax.add_collection3d(poly_collection)


    # Set the scale for the axis
    ax.auto_scale_xyz(scale, scale, scale)
    
#%% Initialization
os.makedirs(os.path.join(cd,'figures'),exist_ok=True)

#%% Main
for s in scans:
    file=glob.glob(os.path.join(source,f'*{s}*nc'))[0]
    Data=xr.open_dataset(file)
    Data=Data.where(Data.range<rmax,drop=True)
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    draw_turbine_3d(ax, 0,0,0, D, H, -90)
    ax.plot(Data.x,Data.y,Data.z,'.g',markersize=3,alpha=0.25)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_aspect('equal')
    ax.set_axis_off()
    plt.title(f'{s}')
    
    plt.savefig(os.path.join(cd,'figures',f'{s}.png'))
    plt.close()
    
    azi_sel=Data.azimuth.mean(dim="scanID").isel(range=0)
    ele_sel=Data.elevation.mean(dim="scanID").isel(range=0)
    
    print(f'{s}')
    print(f'Azi diff: {np.nanmedian(np.diff(azi_sel)[np.diff(azi_sel)>0.25])} deg')
    print(f'Ele diff: {np.nanmedian(np.diff(ele_sel)[np.diff(ele_sel)>0.25])} deg')
    print(f'Min azi: {np.nanmin(azi_sel)} deg')
    print(f'Max azi: {np.nanmax(azi_sel)} deg')
    print(f'Min ele: {np.nanmin(ele_sel)} deg')
    print(f'Max ele: {np.nanmax(ele_sel)} deg')
    print(f'Duration: {np.float64(Data.time.isel(scanID=0,beamID=-1,range=0)-Data.time.isel(scanID=0,beamID=0,range=0))/10**9}')
    print(f'Reps: {len(Data.scanID)}')