#!/usr/bin/env python
# ----------------------------------------------------------------------
#
# Python script to get borehole coordinate info in different coordinate systems.
#
# Charles A. Williams, GNS Science
#
# ----------------------------------------------------------------------
import numpy as np
from pyproj import Transformer

# Filenames.
wgsCoordFile = 'borehole-coords-wgs84.txt'
tmCoordFile = 'borehole-coords-tm.txt'
sampleFile = 'borehole-paraview-sample.txt'

# Coordinate systems.
WGS84 = "EPSG:4326"
TM = "+proj=tmerc +lon_0=178.5 +lat_0=-38.7 +ellps=WGS84 +datum=WGS84 +k=0.9996 +towgs84=0.0,0.0,0.0 +type=crs"
transWGS84ToTM = Transformer.from_crs(WGS84, TM, always_xy=True)

# Get coordinates, convert them, and output them.
coordsWGS84 = np.loadtxt(wgsCoordFile, dtype=np.float64)
(xTM, yTM, zTM) = transWGS84ToTM.transform(coordsWGS84[:,0], coordsWGS84[:,1], coordsWGS84[:,2])
coordsTM = np.column_stack((xTM, yTM, zTM))
np.savetxt(tmCoordFile, coordsTM)

# Compute origin and normal for Paraview profiles.
dx = xTM[1] - xTM[0]
dy = yTM[1] - yTM[0]
normal = np.array([dy, -dx, 0.0])
origin = coordsTM[0,:]
sampleOut = np.vstack((origin, normal))
np.savetxt(sampleFile, sampleOut)

