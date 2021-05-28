#!/usr/bin/env python
# -*- coding: utf8 -*

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, maskoceans, interp

from metos3dutil.plot.plot import Plot
import metos3dutil.metos3d.constants as Metos3d_Constants


class SurfacePlot(Plot):
    """
    Plot of a tracer concentration
    """

    def __init__(self, cmap=None, orientation='lc1', fontsize=8):
        """
        Initializes a surface plot of the ocean

        Parameters
        ----------
        cmap : matplotlib.colors.Colormap or None, default: None
            Colormap used in the plots
        orientation : str, default: 'lc1'
            Orientation of the plot
        fontsize : int, default: 8
            Fontsize used in the plots

        Attributes
        ----------
        _xx : numpy.ndarray
            Numpy array with the used longitudes
        _yy : numpy.ndarray
            Numpy array with the used laditdes
        """
        Plot.__init__(self, cmap=cmap, orientation=orientation, fontsize=fontsize)

        #Longitude
        self._xx = np.concatenate([np.arange(-180 + 0.5 * Metos3d_Constants.METOS3D_GRID_LONGITUDE, 0, Metos3d_Constants.METOS3D_GRID_LONGITUDE), np.arange(0 + 0.5 * Metos3d_Constants.METOS3D_GRID_LONGITUDE, 180, Metos3d_Constants.METOS3D_GRID_LONGITUDE)])
        #Latitude
        self._yy = np.arange(-90 + 0.5 * Metos3d_Constants.METOS3D_GRID_LATITUDE, 90, Metos3d_Constants.METOS3D_GRID_LATITUDE)


    def plot_surface(self, v1d, depth=0, projection='cyl', linewidth=0.15, expandFactor=2, refinementFactor=12, levels=20, vmin=None, vmax=None, pad=0.05, format=None, ticks=None, extend='neither', extendfrac=None, shrink=1.0, clim=None, meridians=None, colorbar=True):
        """
        Plot the tracer concentration for one layer

        Parameters
        ----------
        v1d : numpy.ndarry
            Tracer concentration vector
        depth : int, default: 0
            Layer to plot the tracer concentation
        projection : {'cyl', 'robin'}, default: 'robin'
            Map projection
        linewidth : float, default: 0.15
            Linewidth sets the line widths in pixels for coast lines
        expandFactor : int, default: 2
            Number of steps to extend the tracer concentrations towards the
            coast lines in order to improve accuracy at coast lines (fill the
            whole ocean)
        refinementFactor : int, default: 12
            Refinement of the resolution to improve accuracy at coast lines
        levels : int, default: 20
            Number of levels in the colorbar
        vmin : float or None, default: None
            Minimum value in the figure for the tracer concentration
        vmax : float or None, default: None
            Maximum value in the figure for the tracer concentration
        pad : float, default: 0.05
            Separation between the axes and the color bar in percent of the
            parent axis
        format : str or None, default: None
            Format of the color bar labels
        ticks : list [float] or None, default: None
            Color bar ticks
        extend : {'neither', 'both', 'min', 'max'}
            Determines the contourf-coloring of values that are outside the
            levels range
        extendfrac : {'None, 'auto', length, lengths}, default: None
            Length of the triangular color bar extensions
        shrink : float, default: 1.0
            Fraction by which to multiply the size of the colorbar
        clim : list [float] or None, default: None
            Color limits or the image
        meridians : list [float] or None
            List of meridians to plot
        colorbar : bool, default: True
            If True, plot the color bar

        Returns
        -------
        matplotlib.QuadContourSet
            Returns the result of the function matplotlib.contourf
        """
        assert type(v1d) is np.ndarray and np.shape(v1d) == (Metos3d_Constants.METOS3D_VECTOR_LEN,)
        assert type(depth) is int and 0 <= depth and depth <= Metos3d_Constants.METOS3D_GRID_DEPTH
        assert projection in ['cyl', 'robin']
        assert type(linewidth) is float and 0.0 < linewidth
        assert type(expandFactor) is int and 0 <= expandFactor
        assert type(refinementFactor) is int and 1 <= refinementFactor
        assert levels is None or type(levels) is int and 1 <= levels
        assert type(pad) is float and 0.0 < pad
        assert format is None or type(format) is str
        assert extend in ['neither', 'both', 'min', 'max']
        assert meridians is None or type(meridians) is list
        assert type(colorbar) is bool

        #Prepare data for the surface plot
        v3d = self._reorganize_data(v1d)
        vv = v3d[0,:,:,depth]
        vv = np.roll(vv, 64, axis=1)

        if levels is None:
            vmin = np.nanmin(v3d[0,:,:,:])
            vmax = np.nanmax(v3d[0,:,:,:])
            vstd = np.nanstd(v3d[0,:,:,:])
            levels = np.arange(vmin, vmax, vstd)
        elif vmin is not None and vmax is None:
            vmax = np.nanmax(v3d[0,:,:,:])
            levels = np.linspace(vmin, vmax, num=levels)
        elif vmin is None and vmax is not None:
            vmin = np.nanmin(v3d[0,:,:,:])
            levels = np.linspace(vmin, vmax, num=levels)
        elif vmin is not None and vmax is not None:
            levels = np.linspace(vmin, vmax, num=levels)

        if projection == 'cyl':
            #Basemap
            m = Basemap(projection="cyl", ax=self._axesResult)
            m.drawcoastlines(linewidth=linewidth)

            if meridians is not None:
                m.drawmeridians(meridians, linewidth=linewidth, dashes=[10,10])

            #Xticks
            plt.xticks(range(-180, 181, 45), range(-180, 181, 45))
            plt.xlim([-180, 180])
            self._axesResult.set_xlabel("Longitude [\si{\degree}]")
            #Yticks
            plt.yticks(range(-90, 91, 30), range(-90, 91, 30))
            plt.ylim([-90, 90])
            self._axesResult.set_ylabel("Latitude [\si{\degree}]")

            #TODO mpf: Plot data with colorbar

            self.set_subplot_adjust(left=0.12, bottom=0.19, right=0.915, top=0.99)

        elif projection == 'robin':
            vExpand = self._expandOceanData(vv)
            for _ in range(expandFactor):
                vExpand = self._expandOceanData(vExpand)
            lonsFine, latsFine, vFine = self._refinement(self._xx, self._yy, vExpand, refinementFactor=refinementFactor)
            vFineMask = self._maskOcean(lonsFine, latsFine, vFine)

            #Basemap
            m = Basemap(projection='robin', lon_0=0, resolution='l', ax=self._axesResult)
            m.drawcoastlines(linewidth=linewidth)
            m.drawmapboundary(linewidth=linewidth)
            
            if meridians is not None:
                m.drawmeridians(meridians, linewidth=linewidth, dashes=[10,10])

            xFine, yFine = m(lonsFine, latsFine)
            cntr = m.contourf(xFine, yFine, vFineMask, cmap=self._cmap, levels=levels, origin="lower", extend=extend, vmin=vmin, vmax=vmax)

            if colorbar:
                cbar = m.colorbar(cntr, location='bottom', format=format, ticks=ticks, pad=pad, extend=extend, extendfrac=extendfrac, shrink=0.5)

            #if clim is not None:
            #    plt.clim(clim[0], clim[1])

            #self.set_subplot_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99)

        return cntr


    def plot_slice(self, v1d, slicenum, yvisible=True, yl=6, yr=58, levels=20, vmin=None, vmax=None, cbarLocation='bottom', pad=0.05, format=None, ticks=None, extend='neither', extendfrac=None, colorbar=True):
        """
        Plot the tracer concentration for a slice

        Parameters
        ----------
        v1d : numpy.ndarry
            Tracer concentration vector
        slicenum : int
            Index of the slice to plot (between 0 and 128)
        yvisible : bool, default: True
            If True, create label of the y axis
        yl : int, default: 6
            Latitude constraint index (first index)
        yr : int, default: 58
            Latitude constraint index (second index)
        levels : int, default: 20
            Number of levels in the colorbar
        vmin : float or None, default: None
            Minimum value in the figure for the tracer concentration
        vmax : float or None, default: None
            Maximum value in the figure for the tracer concentration
        cbarLocation : str or int
            Location of the color bar
        pad : float, default: 0.05
            Separation between the axes and the color bar in percent of the
            parent axis
        format : str or None, default: None
            Format of the color bar labels
        ticks : list [float] or None, default: None
            Color bar ticks
        extend : {'neither', 'both', 'min', 'max'}
            Determines the contourf-coloring of values that are outside the
            levels range
        extendfrac : {'None, 'auto', length, lengths}, default: None
            Length of the triangular color bar extensions
        colorbar : bool, default: True
            If True, plot the color bar

        Returns
        -------
        matplotlib.QuadContourSet
            Returns the result of the function matplotlib.contourf
        @author: Markus Pfeil
        """
        assert type(v1d) is np.ndarray and np.shape(v1d) == (Metos3d_Constants.METOS3D_VECTOR_LEN,)
        assert type(slicenum) is int and 0 <= slicenum and slicenum < 128
        assert type(yvisible) is bool
        assert type(yl) is int and 0 <= yl and yl < 64
        assert type(yr) is int and yl <= yr and yr < 64
        assert levels is None or type(levels) is int and 1 <= levels
        assert type(pad) is float and 0.0 < pad
        assert format is None or type(format) is str
        assert extend in ['neither', 'both', 'min', 'max']
        assert type(colorbar) is bool

        v3d = self._reorganize_data(v1d)
        vv = v3d[0, yl:yr, slicenum, :]

        #Depth, heights, mids
        vzz = np.nanmax(v3d[1, :, slicenum, :], axis = 0)
        vdz = np.nanmax(v3d[2, :, slicenum, :], axis = 0)
        vdzz = vzz - 0.5*vdz

        if levels is None:
            vmin = np.nanmin(v3d[0,:,:,:])
            vmax = np.nanmax(v3d[0,:,:,:])
            vstd = np.nanstd(v3d[0,:,:,:])
            levels = np.arange(vmin, vmax, vstd)
        elif vmin is not None and vmax is None:
            vmax = np.nanmax(v3d[0,:,:,:])
            levels = np.linspace(vmin, vmax, num=levels)
        elif vmin is None and vmax is not None:
            vmin = np.nanmin(v3d[0,:,:,:])
            levels = np.linspace(vmin, vmax, num=levels)
        elif vmin is not None and vmax is not None:
            levels = np.linspace(vmin, vmax, num=levels)

        #Plot slice
        cntr = self._axesResult.contourf(self._yy[yl:yr], vdzz, vv.T, cmap=self._cmap, levels=levels, origin="upper")
        
        if colorbar:
            cbar = self._fig.colorbar(cntr, format=format, ticks=ticks, pad=pad, extend=extend, extendfrac=extendfrac)

        #Set xticks
        plt.xticks([-60, -30, 0, 30, 60], [-60, -30, 0, 30, 60])
        self._axesResult.set_xlabel(r'Latitude [\si{\degree}]')

        #Set yticks
        self._axesResult.set_ylim([0, 5200])
        plt.yticks(vzz, ("50", "", "", "360", "", "790", "1080", "1420", "1810", "2250", "2740", "3280", "3870", "4510", "5200"))
        if yvisible:
            self._axesResult.set_ylabel(r'Depth [\si{\metre}]')

        #Invert y axis
        self._axesResult.invert_yaxis()
        #plt.gca().invert_yaxis()
        
        #self.set_subplot_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99)

        return cntr


    def _refinement(self, lons, lats, v, refinementFactor=2):
        """
        Returns a refinement of the data

        Parameters
        ----------
        lons : numpy.ndarray
            Array of longitudes
        lats : numpy.ndarray
            Array of latitudes
        v : numpy.ndarray
            3D tracer concentration vector
        refinementFactor : int, default: 2
            Refinement factor

        Returns
        -------
        tuple [numpy.ndarray]
            Tuple with a
              - numpy array including the refined longitudes,
              - numpy array including the refined latitudes,
              - numpy array including the refined tracer concentration
        """
        xFine = np.linspace(lons[0], lons[-1], lons.shape[0]*refinementFactor)
        yFine = np.linspace(lats[0], lats[-1], lats.shape[0]*refinementFactor)
        lonsFine, latsFine = np.meshgrid(xFine, yFine)

        vFine = interp(v, lons, lats, lonsFine, latsFine, order=1)

        return (lonsFine, latsFine, vFine)


    def _maskOcean(self, lons, lats, v):
        """
        Mask the ocean data

        Parameters
        ----------
        lons : numpy.ndarray
            Array of longitudes
        lats : numpy.ndarray
            Array of latitudes
        v : numpy.ndarray
            3D tracer concentration vector

        Returns
        -------
        numpy.ndarray
            Numpy array with the ocean markers
        """
        m = Basemap(projection='cyl', lon_0=0, resolution='h')
        x, y = m(lons, lats)

        mdata = maskoceans(x, y, v, resolution='h', grid=1.25, inlands=False)
        mask = ~mdata.mask

        return np.ma.array(v, mask=mask)


    def _expandOceanData(self, v):
        """
        Expand the data at the coast line in order to recieve a nicer plot

        Parameters
        ----------
        v : numpy.ndarray
            Tracer concentration vector

        Returns
        -------
        numpy.ndarray
            Expanded tracer concentration vector
        """
        assert type(v) is np.ndarray

        vExpand = np.zeros(shape=v.shape)
        vExpand[:,:] = np.nan

        assert len(v.shape) == 2

        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                if np.isnan(v[i,j]):
                    if (29 <= j and j <= 31 and 39 <= i and i <= 42) or (29 <= j and j <= 33 and i == 42):
                        #Not in the gulf of mexico
                        pass
                    elif (0 <= j and j <= 9 and 56 <= i):
                        # Not in the artic
                        pass
                    elif (30 <= j and j <= 39 and 52 <= i and i <= 53) or (30 <= j and j <= 37 and 54 <= i and i <= 56):
                        #Not in the hudsonbay
                        pass
                    elif (63 <= j and j <= 64 and 43 <= i and i <= 45):
                        #Not in the mittelmeer
                        pass
                    elif (68 <= j and j <= 70 and 50 <= i and i <= 53) or (70 <= j and j <= 74 and 54 <= i and i <= 55):
                        #Not in the baltic sea
                        pass
                    elif (79 <= j and j <= 80 and 35 <= i and i <= 37):
                        #Not in the gulf of Aden
                        pass
                    elif (106 <= j and j <= 112 and 45 <= i and i <= 47):
                        #Not in north of japan
                        pass
                    elif (100 <= j and j <= 101 and 35 <= i and i <= 36):
                        #Not in the gulf of thailand
                        pass
                    #Expand for every latitude
                    elif i+1 < v.shape[0] and not np.isnan(v[i+1,j]):
                        vExpand[i,j] = v[i+1,j]
                    elif 0 < i and i < v.shape[0]-3 and not np.isnan(v[i-1,j]):
                        vExpand[i,j] = v[i-1,j]
                    #Expand for every longitude
                    elif j+1 < v.shape[0] and not np.isnan(v[i,j+1]):
                        vExpand[i,j] = v[i,j+1]
                    elif 0 < j and not np.isnan(v[i,j-1]):
                        vExpand[i,j] = v[i,j-1]
                else:
                    vExpand[i,j] = v[i,j]

        return vExpand

