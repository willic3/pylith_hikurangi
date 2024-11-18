#!/usr/bin/env nemesis

"""
Python script to resample SimpleDB with LWD data to SimpleGridDB
"""

import numpy as np
from numpy import genfromtxt
import math
import scipy
import h5py
from scipy.interpolate import griddata
from pylith.meshio.Xdmf import Xdmf
#from spatialdata.spatialdb.SimpleDB import SimpleDB
from spatialdata.spatialdb.SimpleGridDB import SimpleGridDB
#from spatialdata.spatialdb.SimpleIOAscii import createWriter
from spatialdata.spatialdb.SimpleGridAscii import createWriter
from coordsys_pylith3 import cs_gisborne_mesh
from coordsys_pylith3 import cs_profile2d

# Input/output files.
outProfile2D = 'resampled_sdb.h5'
outSpatialdb = 'resampled_sdb.spatialdb'

# Reference point (trench) in 3D TM coordinates and points defining profile.
refCoordTM = np.array([3.9114289e+04, -2.0573137e+04], dtype=np.float64)
profileCoordsTM = np.array([[3.436441782295286976e+04, -1.771862404823477482e+04],
                            [9.946172787998057174e+03, -3.044082226190067104e+03]], dtype=np.float64)

# Sampling points for 2D spatialdb.
xSample2D = np.array([-5.0e5, -4.0e5, -3.0e5, -2.5e5, -2.0e5, -1.75e5, -1.5e5, -1.4e5, -1.3e5, -1.2e5, -1.1e5, -1.0e5,
                      -95000.0, -90000.0, -85000.0, -80000.0, -75000.0, -70000.0, -65000.0, -60000.0, -55000.0,
                      -50000., -47500., -45000., -42500., -40000., -39800., -39600., -39400., -39200., -39000.,
                      -38800., -38600., -38400., -38200., -38000., -37800., -37600., -37400., -37200., -37000.,
                      -36800., -36600., -36400., -36200., -36000., -35800., -35600., -35400., -35200., -35000.,
                      -34800., -34600., -34400., -34200., -34000., -33800., -33600., -33400., -33200., -33000.,
                      -32800., -32600., -32400., -32200., -32000., -31800., -31600., -31400., -31200., -31000.,
                      -30800., -30600., -30400., -30200., -30000., -29800., -29600., -29400., -29200., -29000.,
                      -28800., -28600., -28400., -28200., -28000., -27800., -27600., -27400., -27200., -27000.,
                      -26800., -26600., -26400., -26200., -26000., -25800., -25600., -25400., -25200., -25000.,
                      -24800., -24600., -24400., -24200., -24000., -23800., -23600., -23400., -23200., -23000.,
                      -22800., -22600., -22400., -22200., -22000., -21800., -21600., -21400., -21200., -21000.,
                      -20800., -20600., -20400., -20200., -20000., -19800., -19600., -19400., -19200., -19000.,
                      -18800., -18600., -18400., -18200., -18000., -17800., -17600., -17400., -17200., -17000.,
                      -16800., -16600., -16400., -16200., -16000., -15800., -15600., -15400., -15200., -15000.,
                      -14800., -14600., -14400., -14200., -14000., -13800., -13600., -13400., -13200., -13000.,
                      -12800., -12600., -12400., -12200., -12000., -11800., -11600., -11400., -11200., -11000.,
                      -10800., -10600., -10400., -10200., -10000., -9800., -9600., -9400., -9200., -9000.,
                      -8800., -8600., -8400., -8200., -8000., -7800., -7600., -7400., -7200., -7000.,
                      -6800., -6600., -6400., -6200., -6000., -5800., -5600., -5400., -5200., -5000.,
                      -4800., -4600., -4400., -4200., -4000., -3800., -3600., -3400., -3200., -3000.,
                      -2800., -2600., -2400., -2200., -2000., -1800., -1600., -1400., -1200., -1000.,
                      -800., -600., -400., -200., -0., 200., 400., 600., 800., 1000.,
                      2500., 5000., 7500., 10000.,  12500., 15000., 17500.,
                      20000.,  22500.,  25000.,  27500.,  30000.,  32500.,  35000.,
                      37500.,  40000.,  42500.,  45000.,  47500.,  50000.,
                      55000.0, 60000.0, 65000.0, 70000.0, 75000.0, 80000.0, 85000.0, 90000.0, 95000.0,
                      1.0e5, 1.1e5, 1.2e5, 1.3e5, 1.4e5, 1.5e5, 1.75e5, 2.0e5, 2.5e5, 3.0e5, 4.0e5, 5.0e5], dtype=np.float64)

ySample2D = np.array([-750000., -620000., -370000., -275000., -225000., -185000.,
                      -155000., -130000., -105000.,  -85000.,  -65000.,  -55000.,
                      -48000.,  -42000.,  -38000.,  -34000.,  -30000.,  -23000.,
                      -15000., -8000., -7800., -7600., -7400., -7200., -7000.,
                      -6800., -6600., -6400., -6200., -6000., -5800., -5600., -5400., -5200., -5000.,
                      -4800., -4600., -4400., -4200., -4000., -3800., -3600., -3400., -3200., -3000.,
                      -2800., -2600., -2400., -2200., -2000., -1800., -1600., -1400., -1200., -1000.,
                      -800., -600., -400., -200., -0., 200., 400., 600., 800., 1000., 15000.], dtype=np.float64)

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


## Read value of points
data = np.loadtxt('all_points_text.txt',skiprows=1,delimiter=',')
x = data[:,0]
y = data[:,1]
vp = data[:,2]
vs = data[:,3]
density = data[:,4]

grid_vp = griddata((x, y), vp, points2D, method='linear')
grid_vs = griddata((x, y), vs, points2D, method='linear')
grid_density = griddata((x, y), density, points2D, method='linear')

grid_numPoints = points2D.shape[0]

# Write profile spatial database.
writer = createWriter(outSpatialdb)

values_sdb = [{'name': "vp",
           'units': "m/s",
           'data': grid_vp},
          {'name': "vs",
           'units': "m/s",
           'data': grid_vs},
          {'name': "density",
           'units': "kg/m**2",
           'data': grid_density}]

writer.write({'points': points2D,
              'x': xSample2D,
              'y': ySample2D,
              'coordsys': csProfile2D,
              'data_dim': 2,
              'values': values_sdb})

# ------- 2D profile -------- #

# Write results to HDF5 file (2D profile).
h5 = h5py.File(outProfile2D, 'w')
verts = h5.create_dataset("geometry/vertices", data=points2D)

timeStatic = np.zeros(1, dtype=np.float64)
time = h5.create_dataset("time", data=timeStatic.reshape(1,1,1), maxshape=(None, 1, 1))

topo = h5.create_dataset("viz/topology/cells", data=connect, dtype='d')
topo.attrs['cell_dim'] = np.int32(2)

vpH = h5.create_dataset("vertex_fields/vp", data=grid_vp.reshape(1, grid_numPoints, 1), maxshape=(None, grid_numPoints, 1))
vpH.attrs['vector_field_type'] = 'scalar'

vsH = h5.create_dataset("vertex_fields/vs", data=grid_vs.reshape(1, grid_numPoints, 1), maxshape=(None, grid_numPoints, 1))
vsH.attrs['vector_field_type'] = 'scalar'

densityH = h5.create_dataset("vertex_fields/density", data=grid_density.reshape(1, grid_numPoints, 1), maxshape=(None, grid_numPoints, 1))
densityH.attrs['vector_field_type'] = 'scalar'

h5.close()
xdmfWriter = Xdmf()
xdmfWriter.write(outProfile2D)