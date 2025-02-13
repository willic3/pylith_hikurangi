#!/usr/bin/env nemesis

## @file create_design_matrix.py

## @brief Python application to generate a set of forward models to evaluate
## surface displacements and volumetric strain at a set of locations, based
## on PyLith-generated Green's functions. Trapezoidal slip functions are used.

import pdb
import sys
import math
import string
import numpy
import h5py
import scipy.interpolate
import scipy.linalg
import scipy.spatial.distance

from spatialdata.geocoords.CSCart import CSCart
from spatialdata.spatialdb.SimpleIOAscii import createWriter
from pylith.meshio.Xdmf import Xdmf

from pythia.pyre.applications.Script import Script as Application

class MkForward(Application):
    """
    Python application to generate a set of forward models to evaluate
    surface displacements and volumetric strain at a set of locations, based
    on PyLith-generated Green's functions. Trapezoidal slip functions are used.
    """
  
    import pythia.pyre.inventory
    ## Python object for managing MkForward facilities and properties.
    ##
    ## \b Properties
    ## @li \b impulse_input_file PyLith fault impulse HDF5 file.
    ## @li \b response_disp_file PyLith displacement response HDF5 file.
    ## @li \b response_strain_file PyLith strain response HDF5 file(s).
    ## @li \b data_file Input file with data, uncertainties, and data type.
    ## @li \b fault_sort_coord Coordinate used for sorting fault coordinates.
    ## @li \b displacement_scale_factor Scale factor applied to displacement responses.
    ## @li \b strain_scale_factor Scale factor applied to dilatational strain responses.
    ## @li \b spatialdb_scale_factor Scale factor applied to slip in spatial database output.
    ## @li \b trapezoid_width Total width of slip trapezoids including tapers.
    ## @li \b taper_width Width of slip taper.
    ## @li \b slip_centers List of trapezoid centers to compute.
    ## @li \b output_root Root name for output files.
    ##

    impulseInputFile = pythia.pyre.inventory.str("impulse_input_file", default="fault.h5")
    impulseInputFile.meta['tip'] = "PyLith fault impulse HDF5 file."

    responseDispFile = pythia.pyre.inventory.str("response_disp_file", default="response.h5")
    responseDispFile.meta['tip'] = "PyLith displacement response HDF5 file."

    responseStrainFile = pythia.pyre.inventory.str("response_strain_file", default="domain.h5")
    responseStrainFile.meta['tip'] = "PyLith strain response HDF5 file."

    dataFile = pythia.pyre.inventory.str("data_file", default="data.txt")
    dataFile.meta['tip'] = "Input file with data, uncertainties, and data type."

    faultSortCoord = pythia.pyre.inventory.int("fault_sort_coord", default=0)
    faultSortCoord.meta['tip'] = "Coordinate used for sorting fault coordinates."

    displacementScaleFactor = pythia.pyre.inventory.float("displacement_scale_factor", default=1.0)
    displacementScaleFactor.meta['tip'] = "Scale factor applied to displacement responses."

    strainScaleFactor = pythia.pyre.inventory.float("strain_scale_factor", default=1.0)
    strainScaleFactor.meta['tip'] = "Scale factor applied to strain responses."

    spatialdbScaleFactor = pythia.pyre.inventory.float("spatialdb_scale_factor", default=-1.0)
    spatialdbScaleFactor.meta['tip'] = "Scale factor applied to slip in spatial database output."

    trapezoidWidth = pythia.pyre.inventory.float("trapezoid_width", default=10000.0)
    trapezoidWidth.meta['tip'] = "Total width of slip trapezoids including tapers."

    taperWidth = pythia.pyre.inventory.float("taper_width", default=2000.0)
    taperWidth.meta['tip'] = "Width of slip taper."

    slipCenters = pythia.pyre.inventory.list("slip_centers", default=[-5000.0, -3000.0])
    slipCenters.meta['tip'] = "List of trapezoid centers to compute."

    outputRoot = pythia.pyre.inventory.str("output_root", default="output")
    outputRoot.meta['tip'] = "Root name for output files."


    # PUBLIC METHODS /////////////////////////////////////////////////////

    def __init__(self, name="lwd_5km_patch"):
        Application.__init__(self, name)

        self.dataVec = None
        self.dataSig = None
        self.dataCoords = None
        self.siteNames = []
        self.dataTypes = []
        self.siteComponents = []

        self.numDispH = 0
        self.numDispU = 0
        self.numDispV = 0
        self.numDil = 0

        self.numImpulses = 0
        self.numTrapezoids = 0
        self.trapInnerWidth = 0.0
        self.numObs = 0
        self.numSites = 0

        self.impulseCoords = None
        self.impulseVals = None
        self.impulseHead = 'X_impulse\tY_impulse'

        self.designMat = None
        self.trapVals = None
        self.dataPredicted = None

        return


    def main(self):
        # pdb.set_trace()
                                         
        self._readData()
        self._createDesign()
        self._runModels()
        self._writeResults()

        return
  

    # PRIVATE METHODS /////////////////////////////////////////////////////


    def _configure(self):
        """
        Setup members using inventory.
        """
        Application._configure(self)
        self.trapezoidHead = 'X0\tS0\tX1\tS1\tX2\tS2\tX3\tS3\tXc\n'
        self.slipCenters = [float(i) for i in self.slipCenters]
        self.numTrapezoids = len(self.slipCenters)

        self.trapInnerWidth = self.trapezoidWidth - 2.0*self.taperWidth
        if (self.trapInnerWidth < 0.0):
            msg = 'Trapezoid parameters yield negative inner width.'
            raise ValueError(msg)

        return


    def _writeResults(self):
        """
        Function to write impulse and response information to text files.
        """
        print("Writing impulses and responses:")
        sys.stdout.flush()
        responseHead = 'X_Center\t' + '\t'.join(self.siteComponents) + '\n'
        outVals = numpy.append(self.trapVals[:,-1].reshape(self.numTrapezoids, 1), self.dataPredicted, axis=1)
        responseFile = self.outputRoot + '_response.txt'
        r = open(responseFile, 'w')
        r.write(responseHead)
        numpy.savetxt(r, outVals, delimiter='\t')
        r.close()

        trapezoidFile = self.outputRoot + '_trapezoid.txt'
        t = open(trapezoidFile, 'w')
        t.write(self.trapezoidHead)
        numpy.savetxt(t, self.trapVals, fmt='%g', delimiter='\t')
        t.close()

        impulseFile = self.outputRoot + '_impulse.txt'
        i = open(impulseFile, 'w')
        i.write(self.impulseHead)
        outVals = numpy.append(self.impulseCoords, self.impulseVals.transpose(), axis=1)
        numpy.savetxt(i, outVals, delimiter='\t')
        i.close()

        for trapNum in range(self.numTrapezoids):
            self._writeSpatialdb(trapNum)

        return


    def _writeSpatialdb(self, trapNum):
        """
        Write a PyLith spatial database for a given trapezoid.
        """
        numString = '_t' + repr(trapNum).rjust(3, '0')
        dbFile = self.outputRoot + numString + '.spatialdb'
        faultSlip = self.spatialdbScaleFactor*self.impulseVals[trapNum,:]
        faultOpening = numpy.zeros_like(faultSlip)

        cs = CSCart()
        cs.inventory.spaceDim = 2
        cs._configure()
        
        data = {
            "points": self.impulseCoords,
            "coordsys": cs,
            "data_dim": 1,
            "values": [
            {"name": "left-lateral-slip",
             "units": "m",
             "data": faultSlip},
            {"name": "fault-opening",
             "units": "m",
             "data": faultOpening},
            ]}

        io = createWriter(dbFile)
        io.write(data)

        return
        
        
    def _createTrapezoid(self, trapCenter):
        """
        Function to create a trapezoidal slip distribution centered on the given value.
        Note that everything is based on using the fault sort coordinate (generally x).
        """

        coords = self.impulseCoords[:,self.faultSortCoord]

        x = numpy.array([trapCenter - 0.5*self.trapezoidWidth,
                         trapCenter - 0.5*self.trapInnerWidth,
                         trapCenter + 0.5*self.trapInnerWidth,
                         trapCenter + 0.5*self.trapezoidWidth], dtype=numpy.float64)
        if (self.taperWidth > 0.0):
            y = numpy.array([0.0, 1.0, 1.0, 0.0], dtype=numpy.float64)
        else:
            y = numpy.array([1.0, 1.0, 1.0, 1.0], dtype=numpy.float64)

        f = scipy.interpolate.interp1d(x, y, assume_sorted=True)

        slipInds = numpy.where(numpy.logical_and(coords >= x[0], coords <= x[3]))
        coordsUse = coords[slipInds]
        slip = f(coordsUse)
        y = numpy.array([0.0, 1.0, 1.0, 0.0], dtype=numpy.float64)

        slipVec = numpy.zeros(self.numImpulses, dtype=numpy.float64)
        slipVec[slipInds] = slip

        trapVals = numpy.zeros(8, dtype=numpy.float64)
        trapVals[0::2] = x
        trapVals[1::2] = y
        trapVals = numpy.append(trapVals, trapCenter)

        return (trapVals, slipVec)
        
        
    def _runModels(self):
        """
        Function to generate trapezoidal impulses and compute responses.
        """
        print("Running slip models:")
        sys.stdout.flush()

        self.dataPredicted = numpy.zeros((self.numTrapezoids, self.numObs), dtype=numpy.float64)
        self.trapVals = numpy.zeros((self.numTrapezoids, 9), dtype=numpy.float64)
        self.impulseVals = numpy.zeros((self.numTrapezoids, self.numImpulses), dtype=numpy.float64)

        for trapNum in range(self.numTrapezoids):
            (trapVals, slipVec) = self._createTrapezoid(self.slipCenters[trapNum])
            predictedVals = numpy.dot(self.designMat, slipVec)
            self.dataPredicted[trapNum, :] = predictedVals
            self.trapVals[trapNum, :] = trapVals
            self.impulseVals[trapNum, :] = slipVec
            self.impulseHead += '\ttrapezoid_%d' % trapNum

        self.impulseHead += '\n'

        return
    
        
    def _readGF(self):
        """
        Function to read impulses and responses.
        """
        print("  Reading impulse/response files:")
        sys.stdout.flush()

        # Open impulse file and determine which fault vertices were used.
        impulses = h5py.File(self.impulseInputFile, 'r')
        impCoords = impulses['geometry/vertices'][:]
        impVals = impulses['vertex_fields/slip'][:,:,1]
        impInds = numpy.nonzero(impVals != 0.0)
        impCoordsUsed = impCoords[impInds[1]]
        impValsUsed = impVals[impInds[0], impInds[1]]
        numImpulses = impValsUsed.shape[0]
        
        # Sort coordinates along given axis and close impulse file.
        impSort = numpy.argsort(impCoordsUsed[:,self.faultSortCoord])
        print(impSort)
        impCoordsSort = impCoordsUsed[impSort,:]
        impValsSort = impValsUsed[impSort]
        impulses.close()

        # Read response displacement file.
        respDispFile = self.responseDispFile
        if (respDispFile != ''):
            respDisp = h5py.File(respDispFile, 'r')
            respDispCoords = respDisp['geometry/vertices'][:]
            respDispVals = respDisp['vertex_fields/displacement'][:]
            respDispValsSort = respDispVals[impSort,:,:]
            respDisp.close()

        # Read strain response file(s).
        
        numBoreholes = 2
        strainRespFile = self.responseStrainFile
        respDilCoords = numpy.zeros((numBoreholes,2), dtype=numpy.float64)
        respDilVals = numpy.zeros((numImpulses, numBoreholes), dtype=numpy.float64)
        #print(respDilVals)
        #print(numpy.shape(respDilVals))
        
        # index of cells containing borehole strains:
        # U1518: 129647, U1519: 2123
        i=0
        for borehole_index in [2123,129647]:
            if (strainRespFile != ''):
                respStrain = h5py.File(strainRespFile, 'r')
                coords = respStrain['geometry/vertices'][:]
                #print('Top of coords',coords[0:5])
                #print('Shape of coords',numpy.shape(coords))
                cells = numpy.array(respStrain['viz/topology/cells'][borehole_index], dtype=numpy.int32)
                #print('cells',cells)
                #print('Shape of cells',numpy.shape(cells))
                cellCoords = coords[cells, :]
                #print('cell coords',cellCoords)
                cellCenter = numpy.mean(cellCoords, axis=0)
                #print('cell center',cellCenter)
                strainField = respStrain['cell_fields/cauchy_strain'][:,borehole_index,:]
                strainFieldSort = strainField[impSort,:]
                cauchy_0 = strainFieldSort[:,0]
                cauchy_1 = strainFieldSort[:,1]
                cauchy_2 = strainFieldSort[:,3]
                strainVals = (cauchy_0+cauchy_1+cauchy_2)/3
                #print(strainVals)
                #print(numpy.shape(strainVals))
                respStrain.close()

                if i>1:
                    break

                respDilCoords[i,:] = cellCenter
                respDilVals[:,i] = strainVals

                i+=1

        #print('strain coords',respDilCoords)
        #print('strain values',respDilVals)

        return (impCoordsSort, respDispCoords, respDispValsSort, respDilCoords, respDilVals)

    
    def _createDesign(self):
        """
        Function to read Green's functions and create design matrix for inversion.
        """

        print("Creating design matrix:")
        sys.stdout.flush()

        # Read Green's function files.
        (self.impulseCoords, respDispCoords, respDispVals, respDilCoords, respDilVals) = self._readGF()
        self.numImpulses = self.impulseCoords.shape[0]

        self.designMat = numpy.zeros((self.numObs, self.numImpulses), dtype=numpy.float64)

        # Create design matrix by sorting response values.
        colNum = 0
        rowNum = 0
        siteNum = 0
        if (self.numDispH != 0):
            dispHInds = self._findCoords(self.dataCoords[0:self.numDispH,:], respDispCoords)
            dispHVals = numpy.transpose(respDispVals[:,dispHInds,0])
            self.designMat[rowNum:self.numDispH, colNum:colNum + self.numImpulses] = dispHVals
            rowNum += self.numDispH
            siteNum += self.numDispH

        if (self.numDispU != 0):
            dispUInds = self._findCoords(self.dataCoords[siteNum:siteNum + self.numDispU,:], respDispCoords)
            dispUVals = numpy.transpose(respDispVals[:,dispUInds,1])
            self.designMat[rowNum:rowNum + self.numDispU, colNum:colNum + self.numImpulses] = dispUVals
            rowNum += self.numDispU
            siteNum += self.numDispU

        if (self.numDispV != 0):
            dispVInds = self._findCoords(self.dataCoords[siteNum:siteNum + self.numDispV,:], respDispCoords)
            dispVValsX = numpy.transpose(respDispVals[:,dispVInds,0])
            dispVValsY = numpy.transpose(respDispVals[:,dispVInds,1])
            self.designMat[rowNum:rowNum + self.numDispV, colNum:colNum + self.numImpulses] = dispVValsX
            rowNum += self.numDispV
            self.designMat[rowNum:rowNum + self.numDispV, colNum:colNum + self.numImpulses] = dispVValsY
            rowNum += self.numDispV
            siteNum += self.numDispV

        if (self.numDil != 0):
            dilInds = self._findCoords(self.dataCoords[siteNum:siteNum + self.numDil,:], respDilCoords)
            dilVals = numpy.transpose(respDilVals[:,dilInds])
            self.designMat[rowNum:rowNum + self.numDil, colNum:colNum + self.numImpulses] = dilVals

        return


    def _findCoords(self, dataCoords, searchCoords):
        """
        Function to find indices of searchCoords within dataCoords.
        """
        tree = scipy.spatial.cKDTree(searchCoords)
        (distances, inds) = tree.query(dataCoords)

        return inds
  
    
    def _readData(self):
        """
        Function to read data file.
        """

        print("Reading data file:")
        sys.stdout.flush()

        f = open(self.dataFile, 'r')
        lines = f.readlines()
        self.numSites = len(lines) - 1
        f.close()

        dispHSites = []
        dispUSites = []
        dispVSites = []
        dilSites = []
        dispHTypes = []
        dispUTypes = []
        dispVTypes = []
        dilTypes = []
        dispHCoords = []
        dispUCoords = []
        dispVCoords = []
        dilCoords = []
        coords = []

        for siteNum in range(self.numSites):
            line = lines[siteNum + 1]
            lineSplit = line.split()
            site = lineSplit[0]
            x = float(lineSplit[1])
            y = float(lineSplit[2])
            dataType = lineSplit[3]
            if (dataType == 'GPSH'):
                self.numDispH += 1
                dispHSites.append(site)
                dispHTypes.append(dataType)
                dispHCoords.append([x, y])
                self.numObs += 1
            elif (dataType == 'GPSU'):
                self.numDispU += 1
                dispUSites.append(site)
                dispUTypes.append(dataType)
                dispUCoords.append([x, y])
                self.numObs += 1
            elif (dataType == 'GPSV'):
                self.numDispV += 1
                dispVSites.append(site)
                dispVTypes.append(dataType)
                dispVCoords.append([x, y])
                self.numObs += 2
            elif (dataType == 'dilStrain'):
                self.numDil += 1
                dilSites.append(site)
                dilTypes.append(dataType)
                dilCoords.append([x, y])
                self.numObs += 1
            else:
                msg = 'Unknown data type %s.' % dataType
                raise ValueError(msg)

        if (self.numDispH != 0):
            self.siteNames += dispHSites
            dispHComponents = [i + '_X' for i in dispHSites]
            self.siteComponents += dispHComponents
            self.dataTypes += dispHTypes
            coords += dispHCoords

        if (self.numDispU != 0):
            self.siteNames += dispUSites
            dispUComponents = [i + '_Y' for i in dispUSites]
            self.siteComponents += dispUComponents
            self.dataTypes += dispUTypes
            coords += dispUCoords

        if (self.numDispV != 0):
            self.siteNames += dispVSites
            dispVComponentsX = [i + '_X' for i in dispVSites]
            dispVComponentsY = [i + '_Y' for i in dispVSites]
            self.siteComponents += dispVComponentsX
            self.siteComponents += dispVComponentsY
            self.dataTypes += dispVTypes
            coords += dispVCoords

        if (self.numDil != 0):
            self.siteNames += dilSites
            dilComponents = [i + '_dilatation' for i in dilSites]
            self.siteComponents += dilComponents
            self.dataTypes += dilTypes
            coords += dilCoords

        self.dataCoords = numpy.array(coords, dtype=numpy.float64)

        return


# ----------------------------------------------------------------------
if __name__ == '__main__':
    app = MkForward()
    app.run()

# End of file
