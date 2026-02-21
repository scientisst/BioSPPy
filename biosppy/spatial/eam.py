    
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

# local
from .. import utils

def plot_geometry(X, triang, values=None, cmap = plt.get_cmap('gist_rainbow', 200), lims = None, type = 'activation', center = None, vec = None):
    
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
    
    lat_a = abs(latq - latp)
    lat_b = abs(latr - latp)
    
    
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
                    cvs[neighbor[0]] = cv[0]
                    vecs[neighbor[0]] = vec
                    cv_X[neighbor[0]] = mesh.points[neighbor[0]]
                    
               
        found.append(i)
    print()                
    return utils.ReturnTuple((cvs, vecs, cv_X),
                             ('cvs', 'vecs', 'cv_X'))


def interpolator(X, y, X_new, kernel='multiquadric'):
    """Interpolates values at new points using Radial Basis Function (RBF) interpolation.
    
    Parameters
    ----------
    X : np.ndarray
        An NxD array of original data points.
    y : np.ndarray
        An array of values corresponding to each point in X.
    X_new : np.ndarray
        An MxD array of new data points where interpolation is to be performed.
    kernel : str, optional
        The type of RBF kernel to use. Defaults to 'multiquadric'.
        
    Returns
    -------
    y_new : np.ndarray
        An array of interpolated values corresponding to each point in X_new.
    """
        
    
    # remove nans from y
    y = y.ravel()
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    
    y_new = scipy.interpolate.RBFInterpolator(X, y, kernel=kernel, epsilon=1.0)(X_new)
    
    return utils.ReturnTuple((y_new,), 
                             ('y_new',))