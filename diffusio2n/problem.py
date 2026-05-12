from ufl import inner, TestFunction, TrialFunction, Measure, grad, lhs, rhs, dot
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem, assemble_matrix, apply_lifting, set_bc, assemble_vector, create_vector
from dolfinx.cpp.geometry import determine_point_ownership
from dolfinx.fem import Constant, Function, form, assemble_scalar, locate_dofs_topological
from dolfinx.nls.petsc import NewtonSolver
from dolfinx import default_scalar_type
from dolfinx import log
from .parameter_utils import constant, calculate_beta
from .solver_utils import info, start_time, execution_time
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np

class Transient:
    def __init__(self, V, subdomains, boundary_conditions, t, T, num_steps, D, u_n, xdmf, thickness_threshold=0.5, f=None):
        """Constructs transient diffusion problem

        Args:
            V (_type_): function space
            subdomains (_type_): subdomains
            boundary_conditions (dict): boundary conditions list
            t (int): start time (s)
            T (int): final time (s)
            num_steps (int): number of time steps 
            D (_type_): diffusion coefficient - m^2/s
            u_n (_type_): initial condition
            xdmf (_type_): solution file
            f (_type_): concentration generation
        """
        problem_type = "Linear"
        u, v = TrialFunction(V), TestFunction(V)
        dx = Measure("dx", domain=V.mesh, subdomain_data=subdomains)

        uh = Function(V)
        uh.name = "phi"
        uh.interpolate(u_n)
                
        xdmf.write_function(uh, t)

        dt = T / num_steps  # time step size
        
        if f:
            pass
        else:
            f = Constant(V.mesh, PETSc.ScalarType(0))

        a = u * v * dx + dt * D * dot(grad(u), grad(v)) * dx
        L = (u_n + dt * f) * v * dx

        bcs = []
        for condition in boundary_conditions:
            if condition.type == "Dirichlet" or condition.type == "DirichletValue":
                bcs.append(condition.bc)
            else:
                a += condition.bc

        bilinear_form = form(a)
        linear_form = form(L)
        
        A = assemble_matrix(bilinear_form, bcs=bcs)
        A.assemble()
        b = create_vector(linear_form)

        # Solve
        info("\n- Transient problem is assembled.")
        t0 = start_time()
        info("\n- Starting the solution..")

        solver = PETSc.KSP().create(V.mesh.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)
        
        points = np.array([0.0, .0, .0])
        point_ownership_data = determine_point_ownership( V.mesh._cpp_object, points, 1e-10)
        cell = point_ownership_data.dest_cells

        coords = V.tabulate_dof_coordinates()
        dim = V.mesh.topology.dim
        active_coords = coords[:, 0] if dim == 1 else coords[:, 1]

        beta_list = []
        phi_list = []
        mass_gain = []
        oxidation_thickness = []

        for i in range(num_steps):

            if isinstance(t, Constant):
                t.value += dt
            else: 
                t += dt

            # Update the right hand side reusing the initial vector
            with b.localForm() as loc_b:
                loc_b.set(0)
            assemble_vector(b, linear_form)

            # Apply Dirichlet boundary condition to the vector
            apply_lifting(b, [bilinear_form], [bcs])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, bcs)

            # Solve linear problem
            solver.solve(b, uh.x.petsc_vec)
            uh.x.scatter_forward()

            # Update solution at previous time step (u_n)
            u_n.x.array[:] = uh.x.array

            time_val = float(t.value) if isinstance(t, Constant) else t
            beta_list.append([time_val, calculate_beta(V.mesh, u_n)])
            mass_gain.append([time_val, assemble_scalar(form(u_n * dx))])
            phi_list.append([time_val, uh.eval(points, cell)[0]])

            oxidized_indices = np.where(uh.x.array >= thickness_threshold)[0]

            if len(oxidized_indices) > 0:
                # Calculate thickness based on the 'Source' location
                if dim == 1:
                    # In 1D (per your OneDimensionalSetup), source is at x=0
                    # Thickness is the furthest distance reached from 0
                    furthest_point = np.max(active_coords[oxidized_indices])
                    base_point = np.min(active_coords) # usually 0.0
                    current_thickness = furthest_point - base_point
                else:
                    # In 2D (per your GMSH setup), source is at y_max
                    # Thickness is the distance from the top down to the lowest point
                    surface_y = np.max(active_coords)
                    lowest_point = np.min(active_coords[oxidized_indices])
                    current_thickness = surface_y - lowest_point
            else:
                current_thickness = 0.0

            oxidation_thickness.append([time_val, current_thickness])
            # Write solution to file
            xdmf.write_function(uh, t)

            phi_val = uh.eval(points, cell)[0]
            print(f"Iteration: {i:3d} | Time: {time_val:6.2f} h | Thickness: {current_thickness:8.4f} um | Phi at center: {phi_val:6.4f}")

        xdmf.close()

        self.solution = uh
        self.beta = np.array(beta_list)
        self.phi_array = np.array(phi_list)
        self.mass_gain = np.array(mass_gain)
        self.oxidation_thickness = np.array(oxidation_thickness)

        info("- Solution is converged.")
        execution_time(t0)

class TransientCoated:
    def __init__(self, V, subdomains, boundary_conditions, t, T, num_steps, substrate_physics, coating_physics, u_n, xdmf, thickness_threshold=0.5, f=None):
        """Constructs transient diffusion problem for coated case"""

        u, v = TrialFunction(V), TestFunction(V)
        dx = Measure("dx", domain=V.mesh, subdomain_data=subdomains)

        uh = Function(V)
        uh.name = "phi"
        uh.interpolate(u_n)
                
        xdmf.write_function(uh, t)

        D = Function(V)
        D.name = "D"

        substrate_markers = subdomains.find(substrate_physics["marker_id"]) # substrate
        coating_markers = subdomains.find(coating_physics["marker_id"]) # coating

        V.mesh.topology.create_connectivity(V.mesh.topology.dim, V.mesh.topology.dim)
        substrate_dofs = locate_dofs_topological(V, V.mesh.topology.dim, substrate_markers)
        coating_dofs = locate_dofs_topological(V, V.mesh.topology.dim, coating_markers)

        D.x.array[substrate_dofs] = np.full_like(substrate_dofs, substrate_physics["D0"], dtype=default_scalar_type)
        D.x.array[coating_dofs] = np.full_like(coating_dofs, coating_physics["D0"], dtype=default_scalar_type)
        D.x.scatter_forward()

        dt = T / num_steps  # time step size
        
        if f:
            pass
        else:
            f = Constant(V.mesh, PETSc.ScalarType(0))

        a = u * v * dx + dt * D * dot(grad(u), grad(v)) * dx
        L = (u_n + dt * f) * v * dx

        bcs = []
        for condition in boundary_conditions:
            if condition.type == "Dirichlet" or condition.type == "DirichletValue":
                bcs.append(condition.bc)
            else:
                a += condition.bc

        bilinear_form = form(a)
        linear_form = form(L)
        
        A = assemble_matrix(bilinear_form, bcs=bcs)
        A.assemble()
        b = create_vector(linear_form)

        # Solve
        info("\n- Transient problem is assembled.")
        t0 = start_time()
        info("\n- Starting the solution..")

        solver = PETSc.KSP().create(V.mesh.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)
        
        points = np.array([0.0, .0, .0])
        point_ownership_data = determine_point_ownership( V.mesh._cpp_object, points, 1e-10)
        cell = point_ownership_data.dest_cells

        coords = V.tabulate_dof_coordinates()
        dim = V.mesh.topology.dim
        active_coords = coords[:, 0] if dim == 1 else coords[:, 1]

        beta_list = []
        phi_list = []
        mass_gain = []
        oxidation_thickness = []

        for i in range(num_steps):

            t += dt
            
            D_s = substrate_physics['D0'] * np.exp(substrate_physics['k'] * t)
            D_c = coating_physics['D0'] * np.exp(coating_physics['k'] * t)

            D.x.array[substrate_dofs] = np.full_like(substrate_dofs, D_s, dtype=default_scalar_type)
            D.x.array[coating_dofs] = np.full_like(coating_dofs, D_c, dtype=default_scalar_type)
            print(max(D.x.array[:]))

            # Update the right hand side reusing the initial vector
            with b.localForm() as loc_b:
                loc_b.set(0)
            assemble_vector(b, linear_form)

            # Apply Dirichlet boundary condition to the vector
            apply_lifting(b, [bilinear_form], [bcs])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, bcs)

            # Solve linear problem
            solver.solve(b, uh.x.petsc_vec)
            uh.x.scatter_forward()

            # Update solution at previous time step (u_n)
            u_n.x.array[:] = uh.x.array

            time_val = float(t.value) if isinstance(t, Constant) else t
            beta_list.append([time_val, calculate_beta(V.mesh, u_n)])
            mass_gain.append([time_val, assemble_scalar(form(u_n * dx))])
            phi_list.append([time_val, uh.eval(points, cell)[0]])

            oxidized_indices = np.where(uh.x.array >= thickness_threshold)[0]

            if len(oxidized_indices) > 0:
                # Calculate thickness based on the 'Source' location
                if dim == 1:
                    # In 1D (per your OneDimensionalSetup), source is at x=0
                    # Thickness is the furthest distance reached from 0
                    furthest_point = np.max(active_coords[oxidized_indices])
                    base_point = np.min(active_coords) # usually 0.0
                    current_thickness = furthest_point - base_point
                else:
                    # In 2D (per your GMSH setup), source is at y_max
                    # Thickness is the distance from the top down to the lowest point
                    surface_y = np.max(active_coords)
                    lowest_point = np.min(active_coords[oxidized_indices])
                    current_thickness = surface_y - lowest_point
            else:
                current_thickness = 0.0

            oxidation_thickness.append([time_val, current_thickness])

            phi_val = uh.eval(points, cell)[0]
            print(f"Iteration: {i:3d} | Time: {time_val:6.2f} h | Thickness: {current_thickness:8.4f} um | Phi at center: {phi_val:6.4f}")

            # Write solution to file
            xdmf.write_function(uh, t)

        xdmf.close()

        self.solution = uh
        self.beta = np.array(beta_list)
        self.phi_array = np.array(phi_list)
        self.mass_gain = np.array(mass_gain)
        self.oxidation_thickness = np.array(oxidation_thickness)

        info("- Solution is converged.")
        execution_time(t0)
