from dolfinx.fem import functionspace, Function, Constant, locate_dofs_topological
from ufl import TrialFunction, exp
from diffusio2n.boundary_conditions import BoundaryCondition
from diffusio2n.problem import Transient
from dolfinx import default_scalar_type
from diffusio2n.dolfinx_utils import CoatingSubstrate1DSetup
from dolfinx.io import XDMFFile
import numpy as np
import os
import params

# Results directory
path = os.path.dirname(os.path.abspath(__file__))
resultsDir = path + "/ResultsDir/paraview/"
resultsDirFEM = path + "/ResultsDir/"
os.makedirs(resultsDir, exist_ok=True)
xdmfFile = '/N.xdmf'

L_coat=params.H_top 
L_sub=params.H_bottom

degree = 1

O2_concentration = 1.0
top_tag = 11

T = 50 # Last time
num_steps = 200

D0 = 3E-5 * 3600  # (micrometer)^2/h
k_decay = -0.005

# Define initial condition
coating_concentration = 0.0
substrate_concentration = 0.0

data = []
N_array = np.arange(1000,5500,500)

for N in N_array:

    mesh, subdomains, facet_tags = CoatingSubstrate1DSetup(N, L_coat, L_sub)


    V = functionspace(mesh, ("Lagrange", degree))
    u = TrialFunction(V)

    # Define the boundary conditions
    boundary_conditions = [BoundaryCondition("DirichletValue", top_tag, O2_concentration, V, facet_tags, u)]

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
    t = Constant(mesh, default_scalar_type(0.0))

    D = D0 * exp(k_decay * t)

    problem = Transient(V, subdomains, boundary_conditions, t, T, num_steps, D, u_n, xdmf)

    data.append([N, problem.mass_gain[-1][1]])

print(data)

meshFile = f'/N_massgain.txt' 
np.savetxt(resultsDirFEM+meshFile, np.array(data))
