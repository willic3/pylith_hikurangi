#!/usr/bin/env nemesis

"""
Python script to interpolate NZ-wide properties to a profile.
"""

import numpy as np
from numpy import genfromtxt
import pandas as pd
import math
import scipy
import h5py
from pylith.meshio.Xdmf import Xdmf
from spatialdata.spatialdb.SimpleDB import SimpleDB
from spatialdata.spatialdb.SimpleGridDB import SimpleGridDB
from spatialdata.spatialdb.SimpleIOAscii import createWriter
from coordsys_pylith3 import cs_gisborne_mesh
from coordsys_pylith3 import cs_profile2d

# Input/output files.
inSpatialdb = '../../nzwide_velmodel/vlnzw2.3_expanded_rot.spatialdb'
outProfile3D = 'nz_sdb_20m_lwd_3D.h5'
outProfile2D = 'nz_sdb_20m_lwd_2D.h5'
outSpatialdb = 'nz_sdb_20m_lwd.spatialdb'

# Reference point (trench) in 3D TM coordinates and points defining profile.
refCoordTM = np.array([3.9114289e+04, -2.0573137e+04], dtype=np.float64)
profileCoordsTM = np.array([[3.436441782295286976e+04, -1.771862404823477482e+04],
                            [9.946172787998057174e+03, -3.044082226190067104e+03]], dtype=np.float64)

# Sampling points for 2D spatialdb.
xSample2D = np.array([-5.0e5, -4.0e5, -3.0e5, -2.5e5, -2.0e5, -1.75e5, -1.5e5, -1.4e5, -1.3e5, -1.2e5, -1.1e5, -1.0e5,
                      -95000.0, -90000.0, -85000.0, -80000.0, -75000.0, -70000.0, -65000.0, -60000.0, -55000.0,
                      -50000., -47500., -45000., -42500., -40000., -37500., -35000.,
                      -32500., -30000., -27500., -25000., -22500., -20000., -17500.,
                      -15000., -12500., -10000., -7500., -5000., -2500., 0.,
                      2500., 5000., 7500., 10000.,  12500., 15000., 17500.,
                      20000.,  22500.,  25000.,  27500.,  30000.,  32500.,  35000.,
                      37500.,  40000.,  42500.,  45000.,  47500.,  50000.,
                      55000.0, 60000.0, 65000.0, 70000.0, 75000.0, 80000.0, 85000.0, 90000.0, 95000.0,
                      1.0e5, 1.1e5, 1.2e5, 1.3e5, 1.4e5, 1.5e5, 1.75e5, 2.0e5, 2.5e5, 3.0e5, 4.0e5, 5.0e5], dtype=np.float64)

ySample2D = np.array([-750000., -620000., -370000., -275000., -225000., -185000.,
                      -155000., -130000., -105000.,  -85000.,  -65000.,  -55000.,
                      -48000.,  -42000.,  -38000.,  -34000.,  -30000.,  -23000.,
                      -15000.,   -8000.,   -5000.,   -3000.,   -1000.,    1000.,
                      15000.], dtype=np.float64)

(y2D, x2D) = np.meshgrid(ySample2D, xSample2D, indexing='ij')
points2D = np.column_stack((x2D.flatten(), y2D.flatten()))

# Create connectivity for HDF5 output.
numX = xSample2D.shape[0]
numY = ySample2D.shape[0]
numCellsX = numX - 1
numCellsY = numY - 1
numCells = numCellsX*numCellsY
connect = np.zeros((numCells, 4), dtype=np.int64)
cellNum = 0

for cellY in range(numCellsY):
    for cellX in range(numCellsX):
        connect[cellNum,0] = cellX + cellY*numX
        connect[cellNum,1] = connect[cellNum,0] + 1
        connect[cellNum,2] = connect[cellNum,1] + numX
        connect[cellNum,3] = connect[cellNum,2] - 1
        cellNum += 1


# Gisborne SSE mesh coordinate system.
csGisborne = cs_gisborne_mesh()

# 2D output coordinate system.
csProfile2D = cs_profile2d()

# Profile orientation.
dx = profileCoordsTM[0,0] - profileCoordsTM[1,0]
dy = profileCoordsTM[1,1] - profileCoordsTM[1,0]
angle = math.atan2(dy, dx)

# Rotated coordinates.
coordsXRot = math.cos(angle) * points2D[:,0]
coordsYRot = math.sin(angle) * points2D[:,0]

# Shifted coordinates.
coordsXShift = coordsXRot + refCoordTM[0]
coordsYShift = coordsYRot + refCoordTM[1]

# Coordinates in Gisborne mesh coordinate system.
coords3D = np.column_stack((coordsXShift, coordsYShift, points2D[:,1]))
numPoints = coords3D.shape[0]

# NZ-wide spatialdb.
dbNzwide = SimpleGridDB()
dbNzwide.inventory.description = 'NZ-wide velocity model'
dbNzwide.inventory.filename = inSpatialdb
dbNzwide.inventory.queryType = 'linear'
dbNzwide._configure()

# Read values from NZ-wide spatialdb.
dbNzwide.open()
queryData = np.zeros((numPoints, 3), dtype=np.float64)
queryErr = np.zeros((numPoints,), dtype=np.int32)
dbNzwide.setQueryValues(["density", "vs", "vp"])
dbNzwide.multiquery(queryData, queryErr, coords3D, csGisborne)
dbNzwide.close()
density = queryData[:,0]
vs = queryData[:,1] / 1e3
vp = queryData[:,2] / 1e3


## --- Add LWD to vp, vs, density, and points --- ##

## ------ Load vp, vs, density from LWD -------- ##

lwd = genfromtxt('downsampled_lwd_new.csv', delimiter=',',skip_header=1)
new_points = np.column_stack((lwd[:,0].flatten(), lwd[:,1].flatten()))
new_vs = lwd[:,2] / 1e3
new_vp = lwd[:,3] / 1e3
new_density = lwd[:,4]

print(new_vp)
print(new_vs)
print(new_density)

# Combine coordinates from sdb and lwd
combined_points = np.vstack((points2D, new_points))

# Combine old and new data points
combined_vp = np.hstack((vp, new_vp))
combined_vs = np.hstack((vs, new_vs))
combined_density = np.hstack((density, new_density))

# delete points from background db that are within lwd data
index = []
index.insert(0,np.where((combined_points[:,0] == -5.000000e+03) & (combined_points[:,1] == -3.000000e+03))[0][0])
index.insert(0,np.where((combined_points[:,0] == -7.500000e+03) & (combined_points[:,1] == -3.000000e+03))[0][0])
index.insert(0,np.where((combined_points[:,0] == -1.000000e+04) & (combined_points[:,1] == -3.000000e+03))[0][0])
index.insert(0,np.where((combined_points[:,0] == -1.000000e+04) & (combined_points[:,1] == -1.000000e+03))[0][0])
index.insert(0,np.where((combined_points[:,0] == -1.250000e+04) & (combined_points[:,1] == -1.000000e+03))[0][0])
index.insert(0,np.where((combined_points[:,0] == -1.500000e+04) & (combined_points[:,1] == -1.000000e+03))[0][0])
index.insert(0,np.where((combined_points[:,0] == -1.750000e+04) & (combined_points[:,1] == -1.000000e+03))[0][0])
index.insert(0,np.where((combined_points[:,0] == -2.000000e+04) & (combined_points[:,1] == -1.000000e+03))[0][0])
index.insert(0,np.where((combined_points[:,0] == -2.250000e+04) & (combined_points[:,1] == -1.000000e+03))[0][0])
index.insert(0,np.where((combined_points[:,0] == -2.500000e+04) & (combined_points[:,1] == -1.000000e+03))[0][0])
index.insert(0,np.where((combined_points[:,0] == -2.750000e+04) & (combined_points[:,1] == -1.000000e+03))[0][0])
index.insert(0,np.where((combined_points[:,0] == -3.000000e+04) & (combined_points[:,1] == -1.000000e+03))[0][0])
index.insert(0,np.where((combined_points[:,0] == -3.250000e+04) & (combined_points[:,1] == -1.000000e+03))[0][0])
index.insert(0,np.where((combined_points[:,0] == -3.500000e+04) & (combined_points[:,1] == -1.000000e+03))[0][0])

combined_points = np.delete(combined_points,index, axis=0)
combined_vp = np.delete(combined_vp,index)
combined_vs = np.delete(combined_vs,index)
combined_density = np.delete(combined_density,index)
#print(combined_points)

combined_numPoints = combined_points.shape[0]

# Create new connectivity for writing xmf file
tri = scipy.spatial.Delaunay(combined_points)
connectivity = tri.simplices


## ---------------------------------------------- ##

# Write results to HDF5 file (3D profile).
h5 = h5py.File(outProfile3D, 'w')
verts = h5.create_dataset("geometry/vertices", data=coords3D)

timeStatic = np.zeros(1, dtype=np.float64)
time = h5.create_dataset("time", data=timeStatic.reshape(1,1,1), maxshape=(None, 1, 1))

topo = h5.create_dataset("viz/topology/cells", data=connect, dtype='d')
topo.attrs['cell_dim'] = np.int32(2)

vpH = h5.create_dataset("vertex_fields/vp", data=vp.reshape(1, numPoints, 1), maxshape=(None, numPoints, 1))
vpH.attrs['vector_field_type'] = 'scalar'

vsH = h5.create_dataset("vertex_fields/vs", data=vs.reshape(1, numPoints, 1), maxshape=(None, numPoints, 1))
vsH.attrs['vector_field_type'] = 'scalar'

densityH = h5.create_dataset("vertex_fields/density", data=density.reshape(1, numPoints, 1), maxshape=(None, numPoints, 1))
densityH.attrs['vector_field_type'] = 'scalar'

h5.close()
xdmfWriter = Xdmf()
xdmfWriter.write(outProfile3D)

# ------- 2D profile -------- #

# Write results to HDF5 file (2D profile).
h5 = h5py.File(outProfile2D, 'w')
verts = h5.create_dataset("geometry/vertices", data=combined_points)

timeStatic = np.zeros(1, dtype=np.float64)
time = h5.create_dataset("time", data=timeStatic.reshape(1,1,1), maxshape=(None, 1, 1))

topo = h5.create_dataset("viz/topology/cells", data=connectivity, dtype='d')
topo.attrs['cell_dim'] = np.int32(2)

vpH = h5.create_dataset("vertex_fields/vp", data=combined_vp.reshape(1, combined_numPoints, 1), maxshape=(None, combined_numPoints, 1))
vpH.attrs['vector_field_type'] = 'scalar'

vsH = h5.create_dataset("vertex_fields/vs", data=combined_vs.reshape(1, combined_numPoints, 1), maxshape=(None, combined_numPoints, 1))
vsH.attrs['vector_field_type'] = 'scalar'

densityH = h5.create_dataset("vertex_fields/density", data=combined_density.reshape(1, combined_numPoints, 1), maxshape=(None, combined_numPoints, 1))
densityH.attrs['vector_field_type'] = 'scalar'

h5.close()
xdmfWriter = Xdmf()
xdmfWriter.write(outProfile2D)

# Write profile spatial database.
writer = createWriter(outSpatialdb)

values_sdb = [{'name': "vp",
           'units': "km/s",
           'data': combined_vp},
          {'name': "vs",
           'units': "km/s",
           'data': combined_vs},
          {'name': "density",
           'units': "kg/m**3",
           'data': combined_density}]

writer.write({'points': combined_points,
              'coordsys': csProfile2D,
              'data_dim': 2,
              'values': values_sdb})

#print(combined_points)
#print(np.shape(combined_points))
#print(combined_vp)
#print(np.shape(combined_vp))

# export text file with points and vp, vs, density
export = {'x': combined_points[:,0],
        'y': combined_points[:,1],
        'vp': combined_vp,
        'vs': combined_vs,
        'density': combined_density}

export_df = pd.DataFrame(export)

export_df.to_csv('all_points_text.txt',index=False)