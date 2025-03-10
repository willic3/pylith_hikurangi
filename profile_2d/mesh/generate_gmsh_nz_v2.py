#!/usr/bin/env nemesis
"""Generate a tri or quad mesh of a subduction zone vertical profile using Gmsh, making
use of the built-in geometry engine.

Run `generate_gmsh.py --help` to see the command line options.


"""
import numpy as np
import itertools
import gmsh
import math
import h5py
from pylith.meshio.gmsh_utils import (VertexGroup, MaterialGroup, GenerateMesh)

class App(GenerateMesh):
    """
    Application for generating the mesh.
    """

    X_EAST = 4.2e+5
    X_WEST = -4.2e+5        
    Y_BOT = -100.0e+3

    DX_FAULT = 5.0e+2
    DX_OBS = 10.0
    DX_BIAS_FAULT = 1.05
    DX_BIAS_OBS = 1.05

    FILENAME_TOPO = "groundsurf-profile-coords2d.tsv"
    FILENAME_SLAB = "fault-profile-coords2d.tsv"


    def __init__(self):
        """Constructor.
        """
        # Set the cell choices available through command line options
        # with the default cell type `tri` matching the PyLith parameter files.
        self.cell_choices = {
            "default": "tri",
            "choices": ["tri"],
            }
        self.filename = "one_material_refined_boreholes.msh"

    def _create_points_from_file(self, filename, col1, col2):
        coordinates = np.loadtxt(filename, skiprows=1)
        points = []
        for xy in coordinates:
            points.append((xy[col1], xy[col2]))
        return points

    def create_geometry(self):
        """Create geometry.
        """

        ## Slab points
        SLAB_POINTS = self._create_points_from_file(self.FILENAME_SLAB, 0, 1)

        # Sort points
        SLAB_POINTS.sort(key=lambda tup: tup[0]) 
        self.SLAB_WEST = 0
        self.SLAB_EAST = len(SLAB_POINTS)-1

        # Topography points
        TOPO_POINTS_TMP = self._create_points_from_file(self.FILENAME_TOPO, 0, 1)
        
        # Sort points 
        TOPO_POINTS_TMP.sort(key=lambda tup: tup[0]) 
        NUM_TOPO_POINTS = len(TOPO_POINTS_TMP)
        TOPO_POINTS = np.zeros((NUM_TOPO_POINTS+2,2), dtype=np.float64)
        TOPO_POINTS[1:-1, :] = TOPO_POINTS_TMP
        TOPO_POINTS[0] = (self.X_WEST, TOPO_POINTS_TMP[0][1])
        TOPO_POINTS[-1] = (self.X_EAST, SLAB_POINTS[-1][1])
        self.TOPO_WEST = 0
        self.TOPO_TRENCH = 263
        self.TOPO_EAST = len(TOPO_POINTS)-1

        # Create curve for topography/bathymetry
        pts_topo = []
        for x, y in TOPO_POINTS:
            pt = gmsh.model.geo.add_point(x, y, 0.0)
            pts_topo.append(pt)
        c_topo = gmsh.model.geo.add_bspline(pts_topo)
        p_topo_west = pts_topo[self.TOPO_WEST]
        p_topo_east = pts_topo[self.TOPO_EAST]
        p_topo_trench = pts_topo[self.TOPO_TRENCH]
        
        # Create b-spline curve for the slab
        pts_slab = []
        for x, y in SLAB_POINTS:
            pt = gmsh.model.geo.add_point(x, y, 0.0)
            pts_slab.append(pt)
        self.c_slab = gmsh.model.geo.add_bspline(pts_slab + [p_topo_trench])
        self.p_slab_west = pts_slab[self.SLAB_WEST]
        self.p_slab_east = pts_slab[self.SLAB_EAST]

        # Create domain boundary curves
        p_bot_west = gmsh.model.geo.add_point(self.X_WEST, self.Y_BOT, 0.0)
        p_bot_east = gmsh.model.geo.add_point(self.X_EAST, self.Y_BOT, 0.0)
        self.c_west = gmsh.model.geo.add_polyline([p_topo_west, p_bot_west])
        self.c_bot = gmsh.model.geo.add_polyline([p_bot_west, p_bot_east])
        self.c_east = gmsh.model.geo.add_polyline([p_bot_east, p_topo_east])

        # Split topo curve at the trench
        curves = gmsh.model.geo.split_curve(c_topo, [p_topo_trench])
        self.c_topo_west = curves[0]
        self.c_topo_east = curves[1]

        # Create surfaces from bounding curves
        loop = gmsh.model.geo.add_curve_loop([
            self.c_west,
            self.c_bot,
            self.c_east,
            -self.c_topo_east,
            self.c_slab,
            -self.c_slab,
            -self.c_topo_west
            ])
        self.s_slab = gmsh.model.geo.add_plane_surface([loop])

        ## Add in points for U1518 and U1519
        self.p_U1518 = gmsh.model.geo.add_point(-5541.6171, -2849.1, 0.0)
        self.p_U1519 = gmsh.model.geo.add_point(-34030.0862, -1264.0, 0.0)

        # ## Add points to define box around boreholes
        # # U1518 -----------
        # self.p_U1518_ul = gmsh.model.geo.add_point(-5546.6171, -2844.1, 0.0)
        # self.p_U1518_ur = gmsh.model.geo.add_point(-5536.6171, -2844.1, 0.0)
        # self.p_U1518_bl = gmsh.model.geo.add_point(-5546.6171, -2854.1, 0.0)
        # self.p_U1518_br = gmsh.model.geo.add_point(-5536.6171, -2854.1, 0.0)
        # self.c_U1518_west = gmsh.model.geo.add_polyline([self.p_U1518_ul, self.p_U1518_bl])
        # self.c_U1518_bot = gmsh.model.geo.add_polyline([self.p_U1518_bl,self.p_U1518_br])
        # self.c_U1518_east = gmsh.model.geo.add_polyline([self.p_U1518_br, self.p_U1518_ur])
        # self.c_U1518_top = gmsh.model.geo.add_polyline([self.p_U1518_ur, self.p_U1518_ul])

        # # U1519 -----------
        # self.p_U1519_ul = gmsh.model.geo.add_point(-34035.0862, -1259.0, 0.0)
        # self.p_U1519_ur = gmsh.model.geo.add_point(-34025.0862, -1259.0, 0.0)
        # self.p_U1519_bl = gmsh.model.geo.add_point(-34035.0862, -1269.0, 0.0)
        # self.p_U1519_br = gmsh.model.geo.add_point(-34025.0862, -1269.0, 0.0)
        # self.c_U1519_west = gmsh.model.geo.add_polyline([self.p_U1519_ul, self.p_U1519_bl])
        # self.c_U1519_bot = gmsh.model.geo.add_polyline([self.p_U1519_bl,self.p_U1519_br])
        # self.c_U1519_east = gmsh.model.geo.add_polyline([self.p_U1519_br, self.p_U1519_ur])
        # self.c_U1519_top = gmsh.model.geo.add_polyline([self.p_U1519_ur, self.p_U1519_ul])

        # ## Create surface for borehole boxes
        # # U1518 -----------
        # loop = gmsh.model.geo.add_curve_loop([
        #     self.c_U1518_west,
        #     self.c_U1518_bot,
        #     self.c_U1518_east,
        #     self.c_U1518_top,
        #     ])
        # self.s_U1518 = gmsh.model.geo.add_plane_surface([loop])

        # # U1519 -----------
        # loop = gmsh.model.geo.add_curve_loop([
        #     self.c_U1519_west,
        #     self.c_U1519_bot,
        #     self.c_U1519_east,
        #     self.c_U1519_top,
        #     ])
        # self.s_U1519 = gmsh.model.geo.add_plane_surface([loop])

        gmsh.model.geo.synchronize()

    def mark(self):
        """Mark geometry for materials, boundary conditions, faults, etc.

        This method is abstract in the base class and must be implemented.
        """
        # Create material for the whole mesh.
        materials = (
            MaterialGroup(tag=1, entities=[self.s_slab]),
            #MaterialGroup(tag=2, entities=[self.s_U1518]),
            #MaterialGroup(tag=3, entities=[self.s_U1519]),
        )
        for material in materials:
            material.create_physical_group()

        ## The code below creates empty physical groups, but I can't assign mesh elements to them after the fact
        # gmsh.model.addPhysicalGroup(0, [], 2)
        # gmsh.model.setPhysicalName(0, 2, "physical group U1518")
        # gmsh.model.addPhysicalGroup(0, [], 3)
        # gmsh.model.setPhysicalName(0, 3, "physical group U1519")

        # Create physical groups for the boundaries and the fault.
        vertex_groups = (
            VertexGroup(name="groundsurf", tag=10, dim=1, entities=[self.c_topo_west, self.c_topo_east]),
            VertexGroup(name="bndry_west", tag=11, dim=1, entities=[self.c_west]),
            VertexGroup(name="bndry_east", tag=12, dim=1, entities=[self.c_east]),
            VertexGroup(name="bndry_bot", tag=14, dim=1, entities=[self.c_bot]),
            VertexGroup(name="fault", tag=15, dim=1, entities=[self.c_slab]),
            VertexGroup(name="fault_end", tag=16, dim=0, entities=[self.p_slab_west]),
        )
        for group in vertex_groups:
            group.create_physical_group()

        
    def generate_mesh(self, cell):
        """Generate the mesh.
        """
        # Set discretization size with geometric progression from distance to the fault.
        # We turn off the default sizing methods.
        gmsh.option.set_number("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.set_number("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.set_number("Mesh.MeshSizeExtendFromBoundary", 0)

        # First, we setup a field `field_distance` with the distance from the fault.
        fault_distance = gmsh.model.mesh.field.add("Distance")
        observatory_distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumber(fault_distance, "Sampling", 200)
        gmsh.model.mesh.field.setNumbers(fault_distance, "CurvesList", [self.c_slab])
        gmsh.model.mesh.field.setNumbers(observatory_distance, "PointsList", [self.p_U1518,self.p_U1519])

        # Second, we setup a field `field_size`, which is the mathematical expression
        # for the cell size as a function of the cell size on the fault, the distance from
        # the fault (as given by `field_size`, and the bias factor.
        # The `GenerateMesh` class includes a special function `get_math_progression` 
        # for creating the string with the mathematical function.
        field_size_fault = gmsh.model.mesh.field.add("MathEval")
        math_exp_fault = GenerateMesh.get_math_progression(fault_distance, min_dx=self.DX_FAULT, bias=self.DX_BIAS_FAULT)
        gmsh.model.mesh.field.setString(field_size_fault, "F", math_exp_fault)
        field_size_obs = gmsh.model.mesh.field.add("MathEval")
        math_exp_obs = GenerateMesh.get_math_progression(observatory_distance, min_dx=self.DX_OBS, bias=self.DX_BIAS_OBS)
        gmsh.model.mesh.field.setString(field_size_obs, "F", math_exp_obs)

        # Finally, we use the field `field_size` for the cell size of the mesh.
        gmsh.model.mesh.field.setAsBackgroundMesh(field_size_fault)
        gmsh.model.mesh.field.setAsBackgroundMesh(field_size_obs)

        if cell == "quad":
            gmsh.option.setNumber("Mesh.Algorithm", 8)
            gmsh.model.mesh.generate(2)
            gmsh.model.mesh.recombine()
        else:
            gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize("Laplace2D")


if __name__ == "__main__":
    App().main()


# End of file
