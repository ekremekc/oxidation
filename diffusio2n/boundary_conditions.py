from dolfinx.fem import Function, locate_dofs_topological, dirichletbc
from ufl import inner, TestFunction, Measure

class BoundaryCondition:
    def __init__(self, type, marker, values, V, facet_tags, u):
        self._type = type
        self.marker = marker
        self.values = values
        self.fdim = V.mesh.topology.dim - 1
        v = TestFunction(V)
        ds = Measure("ds", domain=V.mesh, subdomain_data=facet_tags)

        if type == "DirichletValue":
            u_D = Function(V)
            u_D.x.array[:] = values
            facets = facet_tags.find(marker)
            dofs = locate_dofs_topological(V, self.fdim, facets)
            self._bc = dirichletbc(u_D, dofs)
        elif type == "Neumann":
            self._bc = inner(values, v) * ds(marker)
        elif type == "Robin":
            self._bc = values[0] * inner(u - values[1], v) * ds(marker)
        else:
            raise TypeError("Unknown boundary condition: {0:s}".format(type))

    @property
    def bc(self):
        return self._bc

    @property
    def type(self):
        return self._type
