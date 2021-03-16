#!/usr/bin/env python
# -*- coding: utf8 -*

import matplotlib.pyplot as plt
import numpy as np

import metos3dutil.metos3d.constants as Metos3d_Constants
from metos3dutil.plot.surfaceplot import SurfacePlot


def plotSurface(v1d, filename, depth=0, projection='robin', orientation='gmd', fontsize=8, plotSurface=True, plotSlice=False, slicenum=None, vmin=None, vmax=None):
    """
    Create a figure of the tracer concentration in the ocean

    Parameters
    ----------
    v1d : numpy.ndarry
        Tracer concentration vector
    filename : str
        Filename of the figure
    depth : int, default: 0
        Layer to plot the tracer concentation
    projection : {'cyl', 'robin'}, default: 'robin'
        Map projection
    orientation : str, default: 'gmd'
        Orientation of the plot
    fontsize : int, default: 8
        Fontsize used in the plot
    plotSurface : bool, default: True
        If True, create a figure with the tracer concentration for the given
        layer.
    plotSlice : bool, default: False
        If True, create a figure with the tracer concentration for a slice
    slicenum : list [int] or None, default: None
        Index of the slices to create a figure
    vmin : float or None, default: None
        Minimum value in the figure for the tracer concentration
    vmax : float or None, default: None
        Maximum value in the figure for the tracer concentration

    Notes
    -----
    The functions creates a figure for the surface plot and a figure for each
    slice, respectively. The position of the slices are shown as dashed lines
    in the surface plot.
    """
    assert type(v1d) is np.ndarray
    assert type(depth) is int and 0 <= depth and depth <= Metos3d_Constants.METOS3D_GRID_DEPTH
    assert projection in ['cyl', 'robin']
    assert type(orientation) is str
    assert type(fontsize) is int and 0 < fontsize
    assert type(plotSurface) is bool
    assert type(plotSlice) is bool
    assert slicenum is None or (type(slicenum) is list)
    assert vmin is None or type(vmin) is float
    assert vmax is None or type(vmax) is float
    assert vmin is None or vmax is None or vmin < vmax

    surface = SurfacePlot(orientation=orientation, fontsize=fontsize)

    #Create two subplots (one for the surface plot and the other one for the slice plot)
    if plotSurface and plotSlice:
        surface.init_subplot(1, 2, orientation=orientation, gridspec_kw={'width_ratios': [9,5]})

    #Plot the surface concentration
    if plotSurface:
        meridians = None if slicenum is None else [np.mod(Metos3d_Constants.METOS3D_GRID_LONGITUDE * x, 360) for x in slicenum]
        cntr = surface.plot_surface(v1d, depth=depth, projection=projection, levels=50, vmin=vmin, vmax=vmax, ticks=plt.LinearLocator(6), format='%.1e', pad=0.05, extend='max', meridians=meridians, colorbar=False)
        #cntr = surface.plot_surface(v1d, depth=depth, projection=projection, levels=50, vmin=0.0, vmax=0.002, ticks=plt.LinearLocator(6), format='%.1e', pad=0.05, extend='max', clim=(0.0,0.002), meridians=meridians, colorbar=False)
        #cntr = surface.plot_surface(v1d, depth=depth, projection=projection, levels=50, ticks=plt.LinearLocator(6), format='%.1e', pad=0.05, extend='max', clim=(0.0,3.00), meridians=meridians, colorbar=False)

        #Set subplot for the slice plot
        if plotSlice:
            surface.set_subplot(0,1)

    #Plot the slice plan of the concentration
    if plotSlice and slicenum is not None:
        for s in slicenum:
            cntrSlice = surface.plot_slice(v1d, s, levels=50, vmin=0.0, vmax=0.002, ticks=plt.LinearLocator(6), format='%.1e', pad=0.02, extend='max', colorbar=False)

    if not plotSurface:
        cntr = cntrSlice

    #Add the colorbar
    plt.tight_layout(pad=0.05, w_pad=0.15)
    cbar = surface._fig.colorbar(cntr, ax=surface._axes, format='%.1e', ticks=plt.LinearLocator(5), pad=0.02, aspect=40, extend='max', orientation='horizontal', shrink=0.8)

    surface.savefig(filename)
    surface.close_fig()

