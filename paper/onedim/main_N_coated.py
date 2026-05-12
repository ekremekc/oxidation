from dolfinx.fem import functionspace, Function, Constant, locate_dofs_topological
from mpi4py import MPI
from ufl import TrialFunction, exp
from diffusio2n.boundary_conditions import BoundaryCondition
from diffusio2n.problem import TransientCoated
from dolfinx import default_scalar_type
from diffusio2n.dolfinx_utils import CoatingSubstrate1DSetup
from dolfinx.io import XDMFFile
import numpy as np
import params
import os

# Results directory
path = os.path.dirname(os.path.abspath(__file__))
resultsDir = path + "/ResultsDir/paraview/"
resultsDirFEM = path + "/ResultsDir/fem_data/"
os.makedirs(resultsDir, exist_ok=True)
xdmfFile = '/N_coated.xdmf'

L_coat=params.H_top 
L_sub=params.H_bottom

mesh, subdomains, facet_tags = CoatingSubstrate1DSetup(5000, L_coat, L_sub)

degree = 1

V = functionspace(mesh, ("Lagrange", degree))
u = TrialFunction(V)

O2_concentration = 1.0
top_tag = 11
# Define the boundary conditions
boundary_conditions = [BoundaryCondition("DirichletValue", top_tag, O2_concentration, V, facet_tags, u)]

# Define initial condition
coating_concentration = 0.0
substrate_concentration = 0.0

u_n = Function(V)
u_n.name = "u_n"

coating_markers = subdomains.find(1) # coating
substrate_markers = subdomains.find(2) # substrate

mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)
coating_dofs = locate_dofs_topological(V, mesh.topology.dim, coating_markers)
substrate_dofs = locate_dofs_topological(V, mesh.topology.dim, substrate_markers)

u_n.x.array[coating_dofs] = np.full_like(coating_dofs, coating_concentration, dtype=default_scalar_type)
u_n.x.array[substrate_dofs] = np.full_like(substrate_dofs, substrate_concentration, dtype=default_scalar_type)
u_n.x.scatter_forward()

xdmf = XDMFFile(mesh.comm, resultsDir+xdmfFile, "w")
xdmf.write_mesh(mesh)


# Define temporal parameters
t = 0.0
T = 50 # Last time
num_steps = 200

substrate_physics = {'D0': 3e-5*3600, 'k': -0.005, 'marker_id': 2}
coating_physics = {'D0': 2e-7*3600, 'k': -0.003, 'marker_id': 1}

problem = TransientCoated(V, subdomains, boundary_conditions, t, T, num_steps, substrate_physics, coating_physics, u_n, xdmf, thickness_threshold=params.thickness_threshold)

massGainFile = f'/massGain_N_coated.txt' 
np.savetxt(resultsDirFEM+massGainFile, problem.mass_gain)

thicknessFile = f'/thickness_N_coated.txt' 
np.savetxt(resultsDirFEM+thicknessFile, problem.oxidation_thickness)
