from dolfinx.mesh import meshtags, locate_entities, create_interval, create_rectangle
from mpi4py import MPI
import numpy as np

def CoatingSubstrate1DSetup(n_elem, L_coat, L_sub, tag_coat=1, tag_sub=2):
    """
    Builds a 1D mesh with two subdomains: Coating and Substrate.
    
    Args:
        n_elem (int): Total number of elements in the mesh.
        L_coat (float): Thickness of the coating layer (starts at x=0).
        L_sub (float): Thickness of the substrate layer (starts after coating).
        tag_coat (int): Physical tag for the coating subdomain.
        tag_sub (int): Physical tag for the substrate subdomain.
        
    Returns:
        mesh, subdomains, facet_tags
    """
    # 1. Create the total interval [0, L_total]
    x_end = L_coat + L_sub
    mesh = create_interval(MPI.COMM_WORLD, n_elem, [0.0, x_end])
    
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, tdim)

    # 2. Define Subdomains (Cells)
    # Coating: 0 <= x <= L_coat
    # Substrate: L_coat < x <= L_total
    def coating_locator(x):
        return x[0] <= L_coat + 1e-10

    def substrate_locator(x):
        return x[0] >= L_coat - 1e-10

    # Locate cells for each region
    cells_coat = locate_entities(mesh, tdim, coating_locator)
    cells_sub = locate_entities(mesh, tdim, substrate_locator)

    # Create the cell tags (subdomains)
    # Note: We create a single array for all cells to avoid overlapping tag issues
    indices = np.concatenate([cells_coat, cells_sub])
    values = np.concatenate([np.full(len(cells_coat), tag_coat, dtype=np.int32),
                             np.full(len(cells_sub), tag_sub, dtype=np.int32)])
    
    # Sort indices to ensure meshtags are valid
    sorted_indices = np.argsort(indices)
    subdomains = meshtags(mesh, tdim, indices[sorted_indices], values[sorted_indices])

    # 3. Define Boundaries (Facets/Points)
    # Tag 11: Top Surface (x=0)
    # Tag 10: Bottom Wall (x=L_total)
    # Tag 14: Internal Interface (x=L_coat)
    boundaries = [
        (11, lambda x: np.isclose(x[0], 0.0)),
        (10, lambda x: np.isclose(x[0], x_end)),
        (14, lambda x: np.isclose(x[0], L_coat))
    ]

    facet_indices, facet_markers = [], []
    for (marker, locator) in boundaries:
        facets = locate_entities(mesh, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full(len(facets), marker))

    facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
    facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
    
    sorted_facets = np.argsort(facet_indices)
    facet_tags = meshtags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

    return mesh, subdomains, facet_tags

def RectangleSetup(nx, ny, L, h,  x_f = 0.25, a_f = 0.025):
    """ This function builds two dimensional setup.
        For boundaries, Tag 1 specifies left boundary, 
                        Tag 2 specifies right boundary,
                        Tag 3 specifies bottom boundary,
                        Tag 4 specifies top boundary.  

    """
    
    mesh = create_rectangle(MPI.COMM_WORLD, [np.array([0.0, 0.0]), np.array([L, h])],
                        [nx, ny])
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)

    def fl_subdomain_func(x, x_f=x_f,a_f = a_f, eps=1e-16):
        x = x[0]
        return np.logical_and(x_f - a_f - eps <= x, x <= x_f + a_f + eps)
    tdim = mesh.topology.dim
    marked_cells = locate_entities(mesh, tdim, fl_subdomain_func)
    fl = 0
    subdomains = meshtags(mesh, tdim, marked_cells, np.full(len(marked_cells), fl, dtype=np.int32))

    boundaries = [(1, lambda x: np.isclose(x[0], 0)), # Left
                  (2, lambda x: np.isclose(x[0], L)), # Right
                  (3, lambda x: np.isclose(x[1], 0)), # Bottom
                  (4, lambda x: np.isclose(x[1], h))] # Top

    facet_indices, facet_markers = [], []
    fdim = mesh.topology.dim - 1
    for (marker, locator) in boundaries:
        facets = locate_entities(mesh, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full(len(facets), marker))
    facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
    facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tags = meshtags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

    return mesh, subdomains, facet_tags

def broadcast_vector(vector):
    vector = MPI.COMM_WORLD.gather(vector, root=0)
    if vector:
        vector = [j for i in vector for j in i]
    else:
        vector=[]
    vector = MPI.COMM_WORLD.bcast(vector,root=0)
    return vector