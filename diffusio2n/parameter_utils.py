from dolfinx.fem import (
    functionspace,
    Function,
    form,
    Constant,
    assemble_scalar,
    locate_dofs_topological,
    Expression,
    dirichletbc,
    petsc,
)
# from .dolfinx_utils import unroll_dofmap
from dolfinx import default_scalar_type
from petsc4py import PETSc
from ufl import Measure
from mpi4py import MPI
import numpy as np

def constant(mesh, value):
    return Constant(mesh, default_scalar_type(value))


class Q_volumetric:
    def __init__(self, mesh, subdomains, Q_total, tag=2, degree=0):
        """initializes volumetic heat release term

        Args:
            mesh (_type_): mesh
            subdomains (_type_): subdomains
            Q_total (_type_): heat release rate in W
            tag (int, optional): physical tag of Q_total subdomain. Defaults to 2.
            degree (int, optional): basis function degree. Defaults to 0.
        """
        # volumetric Heat release rate field using subdomains which integrates to 1.
        self.tag = tag
        V = functionspace(mesh, ("DG", degree))
        q = Function(V)
        dx = Measure("dx", subdomain_data=subdomains)
        volume_form = form(Constant(mesh, PETSc.ScalarType(1)) * dx(tag))
        V_pcb = MPI.COMM_WORLD.allreduce(assemble_scalar(volume_form), op=MPI.SUM)
        q_tot = Q_total / V_pcb

        subdomain_cells = subdomains.find(tag)
        q.x.array[subdomain_cells] = np.full_like(subdomain_cells, q_tot, dtype=default_scalar_type)
        q.x.scatter_forward()

        self._function = q

    @property
    def function(self):
        return self._function


def Q_volumetric_function(mesh, subdomains, Q_total, tag=2, degree=0):
    # volumetric Heat release rate field using subdomains which integrates to 1.
    V = functionspace(mesh, ("DG", degree))
    q = Function(V)
    dx = Measure("dx", subdomain_data=subdomains)
    volume_form = form(Constant(mesh, PETSc.ScalarType(1)) * dx(tag))
    V_pcb = MPI.COMM_WORLD.allreduce(assemble_scalar(volume_form), op=MPI.SUM)
    q_tot = Q_total / V_pcb

    subdomain_cells = subdomains.find(tag)
    q.x.array[subdomain_cells] = np.full_like(subdomain_cells, q_tot, dtype=default_scalar_type)
    q.x.scatter_forward()

    return q

def calculate_beta(mesh, phi, degree=0):
    # volumetric Heat release rate field using subdomains which integrates to 1.
    V = functionspace(mesh, ("DG", degree))
    q = Function(V)
    dx = Measure("dx")
    phi_form = form(phi * dx)
    volume_form = form(Constant(mesh, PETSc.ScalarType(1)) * dx)
    phi_domain = MPI.COMM_WORLD.allreduce(assemble_scalar(phi_form), op=MPI.SUM) 
    V_domain = MPI.COMM_WORLD.allreduce(assemble_scalar(volume_form), op=MPI.SUM)
    beta = phi_domain/V_domain

    return beta

def gaussianDiffusionCoefficient(mesh, D_edge, D_center):
    degree = 1

    V = functionspace(mesh, ("Lagrange", degree))
    D = Function(V, name="D")

    # Center of the Gaussian (default: geometric center of the mesh bbox)
    xg = mesh.geometry.x
    xmin = xg[:, 0].min(); xmax = xg[:, 0].max()
    ymin = xg[:, 1].min(); ymax = xg[:, 1].max()
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)

    # If 3D, also get cz
    tdim = mesh.topology.dim
    cz = 0.0
    if tdim == 3:
        zmin = xg[:, 2].min(); zmax = xg[:, 2].max()
        cz = 0.5 * (zmin + zmax)

    # --- Choose sigma so the edge is ~ D_edge on the farthest part of the domain ---
    # Let R be the farthest vertex distance to the center; pick sigma so that
    # exp(-R^2/(2 sigma^2)) ~ eps (small), making D(R) ≈ D_edge within ~ relative eps.
    eps = 1e-3  # the "closeness" of edge value at the farthest point
    if tdim == 2:
        R = np.sqrt(np.max((xg[:,0]-cx)**2 + (xg[:,1]-cy)**2))
    else:
        R = np.sqrt(np.max((xg[:,0]-cx)**2 + (xg[:,1]-cy)**2 + (xg[:,2]-cz)**2))

    sigma = R / np.sqrt(2.0 * np.log(1.0/eps))


    def gaussian_field(x):
        # x has shape (tdim, npts)
        dx = x[0] - cx
        dy = x[1] - cy
        if tdim == 2:
            r2 = dx*dx + dy*dy
        else:
            dz = x[2] - cz
            r2 = dx*dx + dy*dy + dz*dz
        val = D_edge + (D_center - D_edge) * np.exp(-r2 / (2.0 * sigma*sigma))
        return val

    D.interpolate(gaussian_field)

    return D

def gaussianDiffusionCoefficient2(mesh, D_edge, sigma):
    degree = 1

    V = functionspace(mesh, ("Lagrange", degree))
    D = Function(V, name="D")

    # Pick center (geometric middle of the domain here)
    xg = mesh.geometry.x
    cx = 0.5*(xg[:,0].min() + xg[:,0].max())
    cy = 0.5*(xg[:,1].min() + xg[:,1].max())

    tdim = mesh.topology.dim
    if tdim == 3:
        cz = 0.5*(xg[:,2].min() + xg[:,2].max())
    else:
        cz = 0.0

    def gaussian_field(x):
        dx = x[0] - cx
        dy = x[1] - cy
        if tdim == 2:
            r2 = dx*dx + dy*dy
        else:
            dz = x[2] - cz
            r2 = dx*dx + dy*dy + dz*dz
        return D_edge * np.exp(-r2 / (2.0 * sigma*sigma))

    D.interpolate(gaussian_field)

    return D

def irregularDiffusionCoefficient(mesh, facet_tags, boundaryTag, D_edge, D_center, sigma):
    import ufl
    tdim = mesh.topology.dim
    fdim = tdim - 1

    V = functionspace(mesh, ("CG", 1))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = 1.0 * v * ufl.dx

    # Dirichlet BC on the exterior boundary
    boundary_facets = facet_tags.find(boundaryTag)
    bdry_dofs = locate_dofs_topological(V, fdim, boundary_facets)
    zero = Function(V); zero.x.array[:] = 0.0
    bc = dirichletbc(zero, bdry_dofs)

    psi = Function(V, name="psi")
    A = petsc.assemble_matrix(form(a), bcs=[bc])
    A.assemble()
    b = petsc.assemble_vector(form(L))
    petsc.apply_lifting(b, [form(a)], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")  # or "hypre" / "gamg" for big problems
    ksp.solve(b, psi.x.petsc_vec)
    psi.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    # Compute psi_max (global) and define normalized t(x)=psi/psi_max
    psi_local_max = np.max(psi.x.array)
    psi_max = mesh.comm.allreduce(psi_local_max, op=MPI.MAX)
    assert psi_max > 0.0, "psi_max is zero; check boundary conditions/domain."

    # Precompute constants for normalized Gaussian G(t)
    c0 = np.exp(-1.0/(2.0*sigma*sigma))   # exp(- (1-0)^2 / (2σ^2))
    den = 1.0 - c0

    # Build D(x) by interpolation
    D = Function(V, name="D")

    # Use a UFL expression to define t(x) = psi/psi_max
    t = psi / float(psi_max)
    G = (ufl.exp(-((1.0 - t)**2) / (2.0 * sigma*sigma)) - c0) / den
    D_expr = D_edge - (D_edge - D_center) * G

    # Interpolate the UFL expression into V
    D.interpolate(Expression(D_expr, V.element.interpolation_points()))

    return D