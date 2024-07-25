#!/usr/bin/env python

"""
Python script to interpolate NZ-wide properties to a profile.
"""

import numpy as np
import math

# Input/output files.
inFaultFile = 'fault-profile-coords3d.tsv'
inGroundsurfFile = 'groundsurf-profile-coords3d.tsv'
outFaultFile = 'fault-profile-coords2d.tsv'
outGroundsurfFile = 'groundsurf-profile-coords2d.tsv'

# Reference point (trench) in 3D TM coordinates and points defining profile.
refCoordTM = np.array([3.9114289e+04, -2.0573137e+04], dtype=np.float64)
profileCoordsTM = np.array([[3.436441782295286976e+04, -1.771862404823477482e+04],
                            [9.946172787998057174e+03, -3.044082226190067104e+03]], dtype=np.float64)

# Profile orientation.
# dx = profileCoordsTM[0,0] - profileCoordsTM[1,0]
# dy = profileCoordsTM[1,1] - profileCoordsTM[1,0]
# angle = math.atan2(dy, dx)

# Fault profile.
faultDat = np.loadtxt(inFaultFile, skiprows=1, dtype=np.float64)
faultX3d = faultDat[:,-3]
faultY3d = faultDat[:,-2]
faultZ3d = faultDat[:,-1]
sortInds = np.argsort(faultX3d)
fault3d = np.column_stack((faultX3d[sortInds], faultY3d[sortInds], faultZ3d[sortInds]))
fault2d = fault3d[:,0:2] - refCoordTM

# Get rotation angle from profile rather than borehole coordinates.
dx = fault2d[-1,0] - fault2d[0,0]
dy = fault2d[-1,1] - fault2d[0,1]
angle = math.atan2(dy, dx)

# Rotation matrix.
ca = math.cos(angle)
sa = math.sin(angle)
rotMat = np.array([[ca, sa], [-sa, ca]], dtype=np.float64)

faultRot = np.dot(fault2d, rotMat.transpose())

# Output fault profile.
faultProf = np.column_stack((faultRot[:,0], fault3d[:,2]))
np.savetxt(outFaultFile, faultProf)

# Ground surface profile.
groundDat = np.loadtxt(inGroundsurfFile, skiprows=1, dtype=np.float64)
groundX3d = groundDat[:,-3]
groundY3d = groundDat[:,-2]
groundZ3d = groundDat[:,-1]
sortInds = np.argsort(groundX3d)
ground3d = np.column_stack((groundX3d[sortInds], groundY3d[sortInds], groundZ3d[sortInds]))
ground2d = ground3d[:,0:2] - refCoordTM

groundRot = np.dot(ground2d, rotMat.transpose())

# Output ground profile.
groundProf = np.column_stack((groundRot[:,0], ground3d[:,2]))
np.savetxt(outGroundsurfFile, groundProf)
