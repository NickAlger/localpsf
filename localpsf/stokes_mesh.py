import numpy as np
import dolfin as dl
import numpy.typing as typ
from dataclasses import dataclass
import meshio
from .filesystem_helpers import localpsf_root


@dataclass(frozen=True)
class StokesMeshes:
    ice_mesh_3d: dl.Mesh
    basal_mesh_3d: dl.Mesh
    basal_mesh_2d: dl.Mesh
    boundary_markers: dl.MeshFunction
    Radius: float
    lam: float


def make_stokes_meshes(
    mesh_type: str='fine',
    r0: float = 0.05,
    sig: float = 0.4,
    valleys: int = 4,
    valley_depth: float = 0.35,
    bump_height: float = 0.2,
    min_thickness: float = 0.08 / 8.,
    avg_thickness: float = 0.2 / 8.,
    theta: float = -np.pi / 2.,
    dilitation: float = 1.e4,
) -> StokesMeshes:
    if mesh_type == 'coarse':
        mfile_name = str(localpsf_root) + "/numerical_examples/stokes/meshes/cylinder_coarse"
        lam = 1e10
    elif mesh_type == 'medium':
        mfile_name = str(localpsf_root) + "/numerical_examples/stokes/meshes/cylinder_medium"
        lam = 1e11
    elif mesh_type == 'fine':
        mfile_name = str(localpsf_root) + "/numerical_examples/stokes/meshes/cylinder_fine"
        lam = 1e12
    else:
        raise RuntimeError('invalid mesh type ' + mesh_type + ', valid types are coarse, medium, fine')

    ice_mesh_3d = dl.Mesh(mfile_name + ".xml")

    max_thickness = avg_thickness + (avg_thickness - min_thickness)
    A_thickness = max_thickness - avg_thickness

    Length = 1.
    Width = 1.
    Length *= 2 * dilitation
    Width *= 2 * dilitation
    Radius = dilitation

    class BasalBoundary(dl.SubDomain):
        def inside(me, x, on_boundary):
            return dl.near(x[2], 0.) and on_boundary

    class BasalBoundarySub(dl.SubDomain):
        def inside(me, x, on_boundary):
            return dl.near(x[2], 0.)

    class TopBoundary(dl.SubDomain):
        def __init__(me, Height):
            me.Height = Height
            dl.SubDomain.__init__(me)

        def inside(me, x, on_boundary):
            return dl.near(x[1], me.Height) and on_boundary

    boundary_markers = dl.MeshFunction("size_t", ice_mesh_3d, mfile_name + "_facet_region.xml")
    boundary_mesh = dl.BoundaryMesh(ice_mesh_3d, "exterior", True)
    basal_mesh_3d = dl.SubMesh(boundary_mesh, BasalBoundarySub())

    coords = ice_mesh_3d.coordinates()
    bcoords = boundary_mesh.coordinates()
    subbcoords = basal_mesh_3d.coordinates()
    coord_sets = [coords, bcoords, subbcoords]

    def topography(r, t):
        zero = np.zeros(r.shape)
        R0 = r0 * np.ones(r.shape)
        return bump_height * np.exp(-(r / sig) ** 2) * (
                1. + valley_depth * np.sin(valleys * t - theta) * np.fmax(zero, (r - R0) / sig))

    def depth(r, t):
        zero = np.zeros(r.shape)
        R0 = r0 * np.ones(r.shape)
        return min_thickness - A_thickness * np.sin(valleys * t - theta) * np.exp(
            -(r / sig) ** 2) * np.fmax(zero, (r - R0) / sig)

    for k in range(len(coord_sets)):
        for i in range(len(coord_sets[k])):
            x, y, z = coord_sets[k][i]
            r = np.sqrt(x ** 2 + y ** 2)
            t = np.arctan2(y, x)
            coord_sets[k][i, 2] = depth(r, t) * z + topography(r, t)
            coord_sets[k][i] *= dilitation

    # ------ generate 2D mesh from 3D boundary subset mesh
    coords = basal_mesh_3d.coordinates()[:, :2]
    cells = [("triangle", basal_mesh_3d.cells())]
    mesh2D = meshio.Mesh(coords, cells)
    mesh2D.write("mesh2D.xml")
    basal_mesh_2d = dl.Mesh("mesh2D.xml")

    return StokesMeshes(
        ice_mesh_3d, basal_mesh_3d, basal_mesh_2d,
        boundary_markers, Radius, lam
    )


