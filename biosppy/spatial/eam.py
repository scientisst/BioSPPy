    
# -*- coding: utf-8 -*-
"""
biosppy.spatial.eam
-------------------

This module provides functions for computing and visualizing electrophysiological activation maps (EAMs)
from electrogram (EGM) data. It includes functions for calculating conduction velocity (CV) using triangulation 
of activation times, as well as interpolation of values across a 3D mesh. 
The module also provides a function for plotting the geometry of the mesh with the computed values using PyVista.

:copyright: (c) 2015-2026 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""
# 3rd party
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pyvista as pv
import itertools

# local
from .. import utils

def plot_geometry(X, triang, values=None, cmap = plt.get_cmap('gist_rainbow', 200), lims = None, type = 'activation', position='PA',center = None, vec = None, save_path = None):
    
    """
    Plots a 3D mesh using PyVista.
    
    Parameters
    ----------
    X : np.ndarray
        An Nx3 array of vertex coordinates.
    triang : np.ndarray
        An Mx3 array of triangle indices.
    values : np.ndarray, optional
        An array of scalar values for coloring the mesh.
    cmap : matplotlib.colors.Colormap, optional
        A colormap for coloring the mesh.
    lims : tuple, optional
        A tuple specifying the min and max limits for truncating the colormap.
    type : str, optional
        A string specifying the type of map (e.g., 'activation', 'voltage', 'cv', 'cv_vec' or any other custom label). Defaults to 'activation'. 
    position : str, optional
        A string specifying the view position ('PA' for posterior-anterior, 'AP' for anterior-posterior). Defaults to 'PA'.
    save_path : str, optional
        A string specifying the path to save the plot as an image. If None, the plot is not saved. Defaults to None.
    center : np.ndarray, optional
        An Nx3 array of coordinates for the center points of the vectors (required if type is 'cv_vec').
    vec : np.ndarray, optional
        An Nx3 array of vector components (required if type is 'cv_vec').
    """

    if lims is not None:
        range = [lims[0], lims[1]]
    else:
        range = [np.min(values), np.max(values)]
        
    if np.shape(triang)[1] == 3:
        triang = np.hstack((np.ones((triang.shape[0], 1)) * 3,triang)).astype(int)
        
    if type == 'activation':
        label = 'Activation Time (ms)'
    elif type == 'voltage':
        label = 'Voltage (mV)'
    elif type == 'cv':
        label = 'Conduction Velocity (m/s)'
    elif type == 'cv_vec':
        label = type
        
    if position == 'PA':
        # flip 180 horizontally around the y-axis
        X[:, 2] = -X[:, 2]
        
        if vec is not None:
            vec[:, 2] = -vec[:, 2]
            center[:, 2] = -center[:, 2]
        
    elif position == 'AP':
        pass
    else:
        raise ValueError("Invalid position. Use 'PA' or 'AP'.")
        
    mesh = pv.PolyData(X, triang)
    
    plotter = pv.Plotter()
    plotter.set_background('white')
    if type == 'cv_vec':
        glyph_points = pv.PolyData(center)
        glyph_points["vectors"] = vec
        arrows = glyph_points.glyph(
        orient="vectors",
        scale=True,
        factor=10.0
        )
        plotter.add_mesh(mesh, color='lightgray', opacity=0.5, scalars = values, cmap=cmap, clim=[range[0], range[1]], scalar_bar_args={'title': label},)
        plotter.add_mesh(arrows, color='red')
    elif values is not None:
        plotter.add_mesh(mesh, scalars=values, cmap=cmap, clim=[range[0], range[1]], scalar_bar_args={'title': label},)
    else:
        plotter.add_mesh(mesh, color='black')
        
    plotter.view_xy()  # Force XY view
    plotter.add_key_event('s', lambda: plotter.screenshot(save_path))
    plotter.show()

    
    
def _conduction_velocity(p, latp, q, latq, r, latr):
    """Computes the conduction velocity vector using triangulation of three points.
    
    Parameters
    ----------
    p : array
        Coordinates of point p (x, y, z).
    latp : float
        Activation time at point p (ms).
    q : array
        Coordinates of point q (x, y, z).
    latq : float
        Activation time at point q (ms).
    r : array
        Coordinates of point r (x, y, z).
    latr : float
        Activation time at point r (ms).
        
    Returns
    -------
    cv : float
        Conduction velocity (m/s).
    vec : array
        Conduction velocity vector (unit vector in the direction of propagation).
        """
        
    # compute arrays of triangle edges
    a = np.array([q[0]-p[0], q[1]-p[1], q[2]-p[2]])
    b = np.array([r[0]-p[0], r[1]-p[1], r[2]-p[2]])
    c = np.array([r[0]-q[0], r[1]-q[1], r[2]-q[2]])
    
    # compute relative activation times
    
    lat_a = latq - latp
    lat_b = latr - latp
    
    
    theta = np.arccos((np.linalg.norm(a)**2 + np.linalg.norm(b)**2 - np.linalg.norm(c)**2) / (2*(np.linalg.norm(a) * np.linalg.norm(b))))
    alpha = np.arctan2(lat_b*np.linalg.norm(a) - lat_a*np.linalg.norm(b)*np.cos(theta), lat_a*np.linalg.norm(b)*np.sin(theta))
    
    # compute conduction velocity (scalar)
    cv = np.linalg.norm(a) * np.cos(alpha) / lat_a
    
    # compute conduction velocity (vector)
    
    n = np.cross(b, a)
    n = n / np.linalg.norm(n)
    
    x_ps = np.cross(n, a) * np.tan(alpha)  
    vec_full = a - x_ps
    vec = vec_full / np.linalg.norm(vec_full)
    
    return utils.ReturnTuple((cv, vec),
                             ('cv', 'vec'))

def cv_triangulation(egmX, lat, min_dist = 2, min_lat = 2, verbose = 1):
    """Computes the conduction velocity (CV) and CV vector at each point in a 3D mesh using triangulation.
    
    Parameters
    ----------
    egmX : np.ndarray
        An Nx3 array of vertex coordinates.
    lat : np.ndarray
        An array of activation times corresponding to each vertex in egmX.
    min_dist : float, optional
        The minimum distance (in mm) between points to be considered for CV calculation. Defaults to 2 mm.
    min_lat : float, optional
        The minimum difference in activation time (in ms) between points to be considered for CV calculation. Defaults to 2 ms.
    verbose : int, optional
        If set to 1, displays a progress bar during the CV calculation. Defaults to 1.
        
    Returns
    -------
    cvs : np.ndarray
        An array of conduction velocities (in m/s) corresponding to each vertex in egmX
    vecs : np.ndarray
        An Nx3 array of conduction velocity vectors corresponding to each vertex in egmX.
    cv_X : np.ndarray
        An Nx3 array of vertex coordinates corresponding to the calculated conduction velocities and vectors.
    """
    
    # triangulate
    mesh = pv.PolyData(egmX)
    mesh = mesh.delaunay_2d()
    
    faces = mesh.faces.reshape((-1,4))[:, 1:4]
    
    found = []
    # nan vector same length as egmX
    cvs = np.full(len(egmX), np.nan)
    # nan vector same shape as egmX
    vecs = np.full((len(egmX), 3), np.nan)
    # nan vector same shape as egmX
    cv_X = np.full((len(egmX), 3), np.nan)
    count = 0
    for i in range(len(mesh.points)):
        
        neighbors = [j for j in faces if i in j]
        
        if verbose:
            print(f"\rCV calculation. Progress: {i+1}/{len(mesh.points)}", end="", flush=True)
            
        for neighbor in neighbors:
            
            # check if triangle has already been used with previous vertex
            if not bool(set(neighbor) & set(found)):
                # check constraints
                delta_pq = lat[neighbor[1]] - lat[neighbor[0]]
                delta_pr = lat[neighbor[2]] - lat[neighbor[0]]
                
                dist_pq = np.linalg.norm(mesh.points[neighbor[1]] - mesh.points[neighbor[0]])
                dist_pr = np.linalg.norm(mesh.points[neighbor[2]] - mesh.points[neighbor[0]])
                
                if abs(delta_pq) > min_lat and abs(delta_pr) > min_lat and dist_pq > min_dist and dist_pr > min_dist:
                    count += 1
                    cv, vec = _conduction_velocity(mesh.points[neighbor[0]], lat[neighbor[0]], mesh.points[neighbor[1]], lat[neighbor[1]], mesh.points[neighbor[2]], lat[neighbor[2]])
                    cvs[neighbor[0]] = cv
                    vecs[neighbor[0]] = vec
                    cv_X[neighbor[0]] = mesh.points[neighbor[0]]
                    
               
        found.append(i)
        
    return utils.ReturnTuple((cvs, vecs, cv_X),
                             ('cvs', 'vecs', 'cv_X'))




def _remove_duplicate_data_points(x, f_x):
    """Removes duplicate data points from the input arrays x and f_x.
    If there are duplicate coordinates in x, the corresponding values in f_x are averaged. 
    If any of the values in f_x are equal to the minimum or maximum value, those values are preserved instead of averaging.
    
    Parameters
    ----------
    x : array
        An NxM array of coordinates.
    f_x : array
        An array of values corresponding to each coordinate in x.
        
    Returns
    -------
    x : array
        An array of unique coordinates.
    f_x : array
        An array of values corresponding to each unique coordinate in x.
    """
    x = np.asarray(x, float)
    f_x = np.asarray(f_x, float)
    _, first, inv = np.unique(x, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    if len(first) == len(x):
        return x, f_x

    fmin, fmax = f_x.min(), f_x.max()
    f_new = f_x.copy()
    counts = np.bincount(inv)
    for g in np.where(counts > 1)[0]:
        idx = np.where(inv == g)[0]
        v = f_x[idx]
        has_min, has_max = v.min() == fmin, v.max() == fmax
        if has_min and has_max:
            val = v.mean()
        elif has_min:
            val = v.min()
        elif has_max:
            val = v.max()
        else:
            val = v.mean()
        f_new[idx[0]] = val          # keep the first occurrence

    keep = np.zeros(len(x), bool)
    keep[first] = True
    return utils.ReturnTuple((x[keep], f_new[keep]), ('x', 'f_x'))


def interpolator(X, y, X_new, kernel='multiquadric', shape_parameter=1.0, distance_threshold=10.0):
    """OpenEP-style RBF interpolation: dedupe -> drop NaNs -> exact RBF -> blank far vertices."""
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()

    m = ~np.isnan(y)                                   # drop NaNs first
    X, y = _remove_duplicate_data_points(X[m], y[m])

    # MATLAB multiquadric is sqrt(r^2 + c^2); SciPy's is sqrt((eps*r)^2 + 1), so eps = 1/c
    rbf = scipy.interpolate.RBFInterpolator(X, y, kernel=kernel, epsilon=1.0 / shape_parameter, smoothing=0.0)
    y_new = rbf(np.asarray(X_new, float))

    d, _ = scipy.spatial.cKDTree(X).query(X_new)
    y_new[d > distance_threshold] = np.nan

    return utils.ReturnTuple((y_new,), ('y_new',))


def _to_o3d(points, voxel_size=None):
    """Convert an (N, 3) array to an Open3D point cloud, optionally voxel-downsampled."""

    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("Open3D is required for registration (pip install open3d).")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if voxel_size:
        pcd = pcd.voxel_down_sample(voxel_size)

    return pcd


def _apply_transform(points, transformation):
    """Apply a 4x4 homogeneous transformation to an array of 3D points.

    Parameters
    ----------
    points : array
        Point coordinates, with shape (N, 3).
    transformation : array
        Homogeneous transformation matrix, with shape (4, 4).

    Returns
    -------
    points : array
        Transformed point coordinates, with shape (N, 3).

    """

    points = np.asarray(points, dtype=float)
    transformation = np.asarray(transformation, dtype=float)
    
    rotation = transformation[:3, :3]     # 3x3 rotation block
    translation = transformation[:3, 3]   # last column: shift in x, y, z

    # rotate each point (one point per row), then shift
    rotated = points @ rotation.T
    
    return utils.ReturnTuple((rotated + translation,), ('points',))



def _plot_registration(eam_points, registered_points, reference_points, path=None):
    """Plot an EAM point cloud before and after registration to a reference.

    Parameters
    ----------
    eam_points : array
        Original EAM point coordinates, with shape (N, 3).
    registered_points : array
        Registered EAM point coordinates, with shape (N, 3).
    reference_points : array
        Reference mesh vertices, with shape (M, 3).
    path : str, optional
        If provided, the figure is saved to this file instead of shown.

    """

    plotter = pv.Plotter(shape=(1, 2), off_screen=path is not None)

    for col, (title, pts) in enumerate((("Before", eam_points), ("After", registered_points))):
        plotter.subplot(0, col)
        plotter.add_text(title, font_size=10)
        plotter.add_mesh(
            pv.PolyData(np.asarray(reference_points, dtype=float)),
            color="lightgray", point_size=3,
        )
        plotter.add_mesh(
            pv.PolyData(np.asarray(pts, dtype=float)),
            color="red", point_size=6,
        )
        plotter.reset_camera()

    plotter.link_views()

    if path is not None:
        plotter.screenshot(path)
        plotter.close()
    else:
        plotter.show()


def register_icp(eam_points, reference_points, voxel_size=2.0, threshold=10.0, permutations=None, flips=None, center=None, show=False, save_path=None):
    """Register EAM points to a reference anatomy.

    Different orientations of the EAM coordinates (axis permutations and axis
    reflections) are explored. Each candidate is centered on the reference and
    refined with point-to-point ICP. The best candidate is the one with the
    highest fitness and, in case of a tie, the lowest inlier RMSE.

    Parameters
    ----------
    eam_points : array
        EAM point coordinates, with shape (N, 3).
    reference_points : array
        Reference mesh vertices (e.g., from a CT segmentation), with shape (M, 3).
    voxel_size : float, optional
        Voxel size used to downsample both point sets before ICP, in the same
        units as the coordinates. If None or 0, no downsampling is performed.
    threshold : float, optional
        Maximum correspondence distance for ICP, in the same units as the
        coordinates.
    permutations : list, optional
        Axis permutations to test. Defaults to all 6 permutations of (0, 1, 2).
    flips : list, optional
        Axis sign combinations to test. Defaults to all 8 combinations of +1/-1,
        which includes reflections.
    center : array, optional
        Point on which the EAM centroid is placed before ICP. Defaults to the
        centroid of `reference_points`.
    show : bool, optional
        If True, a plot of the EAM before and after registration is shown.
    save_path : str, optional
        If provided, the plot is saved to this file instead of shown.
        
    Returns
    -------
    transformation : array
        4x4 matrix mapping the original EAM coordinates to the reference frame
        (orientation, centering and ICP combined). It can be applied to
        full-resolution meshes with `apply_transform`.
    registered_points : array
        `eam_points` after registration, with shape (N, 3).
    permutation : tuple
        Selected axis permutation.
    flip : tuple
        Selected axis signs.
    fitness : float
        ICP fitness of the selected candidate (fraction of inlier correspondences).
    rmse : float
        ICP inlier RMSE of the selected candidate.

    """

    # check inputs
    eam_points = np.asarray(eam_points, dtype=float)
    reference_points = np.asarray(reference_points, dtype=float)

    for name, pts in (("eam_points", eam_points), ("reference_points", reference_points)):
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("`%s` must have shape (N, 3)." % name)

    if permutations is None:
        permutations = list(itertools.permutations(range(3)))

    if flips is None:
        flips = list(itertools.product([1, -1], repeat=3))

    if center is None:
        center = reference_points.mean(axis=0)
    center = np.asarray(center, dtype=float)

    import open3d as o3d

    source = _to_o3d(eam_points, voxel_size)
    target = _to_o3d(reference_points, voxel_size)
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    eam_mean = eam_points.mean(axis=0)

    best_fitness = -1.0
    best_rmse = np.inf
    best_perm = None
    best_flip = None
    best_transform = None

    for perm in permutations:
        for flip in flips:
            # signed permutation: oriented[:, j] = flip[j] * points[:, perm[j]]
            orient = np.zeros((3, 3))
            orient[np.arange(3), perm] = flip

            # initial transform: orient, then center on the reference
            init = np.eye(4)
            init[:3, :3] = orient
            init[:3, 3] = center - orient @ eam_mean

            result = o3d.pipelines.registration.registration_icp(
                source, target, threshold, init, estimation
            )

            if result.fitness > best_fitness or (
                np.isclose(result.fitness, best_fitness)
                and result.inlier_rmse < best_rmse
            ):
                best_fitness = result.fitness
                best_rmse = result.inlier_rmse
                best_perm = tuple(perm)
                best_flip = tuple(flip)
                best_transform = np.asarray(result.transformation)

    registered_points, = _apply_transform(eam_points, best_transform)
    
    if show:
        _plot_registration(eam_points, registered_points, reference_points, path=save_path)

    # output
    args = (best_transform, registered_points, best_perm, best_flip, best_fitness, best_rmse)
    names = ("transformation", "registered_points", "permutation", "flip", "fitness", "rmse")

    return utils.ReturnTuple(args, names)