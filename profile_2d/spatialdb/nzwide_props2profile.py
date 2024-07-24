#!/usr/bin/env python

"""
Python script to interpolate NZ-wide properties to a profile.
"""

import numpy as np
import math
import h5py
from pyproj import Transformer
from pylith.meshio.Xdmf import Xdmf
from spatialdata.spatialdb.SimpleGridDB import SimpleGridDB
from spatialdata.spatialdb.SimpleGridAscii import createWriter
from coordsys_pylith3 import cs_gisborne_mesh
from coordsys_pylith3 import cs_profile2d

# Input/output files.
inSpatialdb = '../../nzwide_velmodel/vlnzw2.3_expanded_rot.spatialdb'
outProfile3D = 'vlnzw2.3_profile3d.h5'
outProfile2D = 'vlnzw2.3_profile2d.h5'
outSpatialdb = 'vlnzw2.3_profile2d.spatialdb'

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
                      1.0e5, 1.1e5, 1.2e5, 1.3e5, 1.4e5, 1.5e5, 1.75e5, 2.0e5, 2.5e5, 3.0e5, 4.0e5], dtype=np.float64)

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
vs = queryData[:,1]
vp = queryData[:,2]

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

# Write results to HDF5 file (2D profile).
h5 = h5py.File(outProfile2D, 'w')
verts = h5.create_dataset("geometry/vertices", data=points2D)

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
xdmfWriter.write(outProfile2D)

# Write profile spatial database.
writer = createWriter(outSpatialdb)

values = [{'name': "vp",
           'units': "m/s",
           'data': vp},
          {'name': "vs",
           'units': "m/s",
           'data': vs},
          {'name': "density",
           'units': "kg/m**2",
           'data': density}]

writer.write({'points': points2D,
              'x': xSample2D,
              'y': ySample2D,
              'coordsys': csProfile2D,
              'data_dim': 2,
              'values': values})

