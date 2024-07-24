#!/usr/bin/env python

# Coordinate systems

from spatialdata.geocoords.CSGeo import CSGeo
from spatialdata.geocoords.CSCart import CSCart
from spatialdata.geocoords.CSGeoLocal import CSGeoLocal

# Geographic lat/lon coordinates in WGS84 datum
def cs_geo():
    """Geographic lat/lon coordinates in WGS84 datum.
    """
    cs = CSGeo()
    cs.crsString = "EPSG:4326"
    # Note:  proj ordering for this CS is lat, lon.
    cs.spaceDim = 2
    cs._configure()
    return cs


def cs_geo3D():
    """Geographic lat/lon/elev coordinates in WGS84 datumm.

    """
    cs = CSGeo()
    cs.crsString = "EPSG:4326"
    # Note:  proj ordering for this CS is lat, lon.
    cs.spaceDim = 3
    cs._configure()
    return cs


# Coordinate system used in all-Hikurangi finite-element meshes
def cs_meshHik():
    cs = CSGeo()
    cs.crsString = "+proj=tmerc +datum=WGS84 +lon_0=175.45 +lat_0=-40.825 +k=0.9996 +units=m +type=crs"
    # Note:  proj ordering for this CS is E, N.
    cs.spaceDim = 3
    cs._configure()
    return cs


# Coordinate system for Gisborne SSE mesh.
def cs_gisborne_mesh():
    cs = CSGeo()
    cs.crsString = "+proj=tmerc +datum=WGS84 +lon_0=178.5 +lat_0=-38.7 +k=0.9996 +units=m +type=crs"
    cs.spaceDim = 3
    cs._configure()
    return cs

# Coordinate system for 2D profile.
def cs_profile2d():
    cs = CSCart()
    cs.spaceDim = 2
    cs._configure()
    return cs


# Coordinate system for rotated Gisborne SSE mesh.
def cs_gisborne_rot_mesh():
    cs = CSGeoLocal()
    cs.crsString = "+proj=tmerc +datum=WGS84 +lon_0=178.5 +lat_0=-38.7 +k=0.9996 +units=m +type=crs"
    cs.spaceDim = 3
    # Need to determine whether originX and originY should be in geographic (178.5, -38.7)
    # or projected coordinates.
    cs.originX = 265244.0746309568
    cs.originY = 231428.15423679753
    cs.yAzimuth = -31.2
    cs._configure()
    return cs


# Coordinate system used by Donna's NZTM-based velocity model.
def cs_nzwide_nztm():
    cs = CSGeoLocal()
    cs.crsString = "+proj=tmerc +lat_0=0 +lon_0=173 +k=0.9996 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs +type=crs"
    # Note:  proj ordering for this CS is E, N.
    cs.spaceDim = 3
    cs.originX = -8004.76292451
    cs.originY = -4623556.23121048
    cs.yAzimuth = 40.0
    cs._configure()
    return cs


def cs_nztm():
    cs = CSGeo()
    cs.crsString = "EPSG:2193"
    # Note:  proj ordering for this CS is N, E.
    cs.spaceDim = 3
    cs._configure()
    return cs


def cs_nzmg49():
    cs = CSGeo()
    cs.crsString = "EPSG:27200"
    cs.spaceDim = 3
    cs._configure()
    return cs


# End of file
