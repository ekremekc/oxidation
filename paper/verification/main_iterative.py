from dolfinx.fem import functionspace, Function
from mpi4py import MPI
from dolfinx import default_scalar_type
from ufl import TrialFunction
from diffusio2n.problem import Transient
from diffusio2n.boundary_conditions import BoundaryCondition
from dolfinx.io import XDMFFile

from diffusio2n.dolfinx_utils import CoatingSubstrate1DSetup
import numpy as np
import os

projectRoot = os.path.join(os.getcwd())
resultsDir = os.path.join(projectRoot,'ResultsDir')
screenshotsDir = os.path.join(resultsDir,'screenshots')
test_name = "transient_iterative"
xdmf_filename = "/"+test_name +".xdmf"

def run(T):
    n_elem = 20
    L_total = 50
    left_concentration = +0.0
    right_concentration = +0.0
    initial_concentration = 1.0
    D = 1

    mesh, subdomains, facet_tags = CoatingSubstrate1DSetup(n_elem, L_coat=L_total, L_sub=0)

    degree = 1

    V = functionspace(mesh, ("CG", degree))
    u = TrialFunction(V)

    xdmf = XDMFFile(V.mesh.comm, resultsDir+xdmf_filename, "w")
    xdmf.write_mesh(V.mesh)

    # define initial condition
    u_n = Function(V)
    u_n.name = "u_n"
    u_n.x.array[:]= default_scalar_type(initial_concentration)


    # Define temporal parameters
    t = 0  # Start time
    num_steps = 100

    # Define the boundary conditions
    boundary_conditions = [
        BoundaryCondition("DirichletValue", 11, left_concentration, V, facet_tags, u),
        BoundaryCondition("DirichletValue", 10, right_concentration, V, facet_tags, u),
    ]

    problem = Transient(V, subdomains, boundary_conditions, t, T, num_steps, D, u_n, xdmf)
    phi = problem.solution
    x_numeric = V.tabulate_dof_coordinates()[:, 0]


    # Analytical solution
    N_inf = 20001
    N = np.arange(1, N_inf , 2)

    def u_t(x, T, D, N, initial_concentration):
        
        def term(x, T, D, n):
            return 1/n*np.exp(-n**2*np.pi**2*D*T/2500)*np.sin(n*np.pi*x/50)
        
        solution = np.zeros(len(x))
        for n in N:
            temp = term(x, T, D, n)
            solution += temp
        solution *= 4*initial_concentration/np.pi
        return solution

    phi_analytic = u_t(x_numeric,T,D,N,initial_concentration)
    phi_numeric = phi.x.array
    return x_numeric, phi_analytic, phi_numeric 

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14}) # Sets a universal default

t_array = [20, 50, 150, 300]

x_numeric, _, _ = run(0.00001)
plt.figure(figsize=(6,4))
for i, T in enumerate(t_array):
    x_numeric, phi_analytic, phi_numeric = run(T)
    plt.plot(x_numeric, phi_analytic, label="t= "+str(T)) # 
    plt.plot(x_numeric, phi_numeric, "rs") # 


plt.xlabel(r"$x$")
plt.ylabel(r"Concentration, $c$ (%)")
plt.tight_layout()
plt.savefig(resultsDir+"/Figure_appendix_verification.png", dpi=600)
plt.legend()
plt.savefig(resultsDir+"/Figure_appendix_verification.pdf", dpi=300)
plt.show()