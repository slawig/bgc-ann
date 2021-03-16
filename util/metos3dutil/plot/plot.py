#!/usr/bin/env python
# -*- coding: utf8 -*

import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import metos3dutil.metos3d.constants as Metos3d_Constants
import metos3dutil.petsc.petscfile as petsc


class Plot():
    """
    Basic functionality for plotting
    """

    def __init__(self, cmap=None, orientation='lc1', fontsize=8, params=None, projection=None):
        """
        Initialization of the plot environment

        Parameters
        ----------
        cmap : matplotlib.colors.Colormap or None, default: None
            Colormap used in the plots
        orientation : str, default: 'lc1'
            Orientation of the plot
        fontsize : int, default: 8
            Fontsize used in the plots
        params : dict or None, default: None
            Parameters for matplotlib
        projection : str or None, default: None
           Projection used for the plot

        Attributes
        ----------
        _colors : dict
            Assignment of a color for each time step
        _cmap : matplotlib.colors.Colormap
            Colormap used in the plots. The default colormap is cm.coolwarm.
        _fig : matplotlib.Figure
            Top level container for all the plot elements
        _axesResult : matplotlib.Axes
            Axes contains most of the figure elements
        """
        assert type(orientation) is str
        assert type(fontsize) is int and 0 < fontsize

        #Colors
        self._colors = {
                -1: 'black', 
                 0: 'C1', 
                 1: 'C0', 
                 2: 'C2', 
                 3: 'C3', 
                 4: 'C4', 
                 5: 'C5', 
                 6: 'C6', 
                 7: 'C7', 
                 8: 'C8', 
                 9: 'C9', 
                10: '#aec7e8', 
                11: '#ffbb78', 
                12: '#98df8a', 
                13: '#ff9896', 
                14: '#c5b0d5', 
                15: '#c49c94', 
                16: '#f7b6d2', 
                17: '#c7c7c7', 
                18: '#dbdb8d', 
                19: '#9edae5'}

        #Colormap
        self._cmap = plt.cm.coolwarm
        if cmap is not None:
            self._cmap = cmap

        self._init_plot(orientation=orientation, fontsize=fontsize, params=params, projection=projection)


    def _init_orientation(self, orientation='lc1'):
        """
        Initializes the orientation (width and height) of the plot

        Parameters
        ----------
        orientation : str, default: 'lc1'
            Orientation of the plot

        Notes
        -----
        If the orientation is not valid, the program is terminated.
        """
        assert type(orientation) is str

        #Orientation
        if orientation == 'lan':
            width = 1.25*11.7*0.7
            height = 0.4*8.3*0.7
        elif orientation == 'ln2':
            width = 1.25*11.7*0.7
            height = 8.3*0.7
        elif orientation == 'lc1':
            width = 5.8078
            height = width / 1.618
        elif orientation == 'lc2':
            #Latex Figure for 2 columns
            width = 2.725
            height = width / 1.618
        elif orientation == 'gmd':
            #Latex Figure for 2 columns paper gmd
            width = 3.26771
            height = width / 1.618
        elif orientation == 'gm2':
            #Latex Figure for 2 columns paper gmd
            width = 3.26771
            height = 1.55
        elif orientation == 'por':
            height = 11.7*0.7
            width = 8.3*0.7
        elif orientation == 'lp1':
            width = 7.5
            height = width / 1.618
        elif orientation == 'lp2':
            width = 1.99
            height = 1.99
        elif orientation == 'etn':
            #Latex Figure for 2 columns paper etna
            width = 2.559
            height = width / 1.618
        elif orientation == 'etnasp':
            width = 5.1389
            height = 2.1
        elif orientation == 'etnatp': 
            #Tatortplot for the paper etna
            width = 1.22
            height = width
        elif orientation == 'etnatp4': 
            #Plot of four tatortplots for the paper etna
            width = 5.1389
            height = 1.22
        else:
            print("Init_Plot: ORIENTATION NOT VALID")
            sys.exit()

        return (width, height)


    def _init_plot(self, orientation='lc1', fontsize=8, params=None, projection=None):
        """
        Initialize the plot windows

        Parameters
        ----------
        orientation : str, default: 'lc1'
            Orientation of the plot
        fontsize : int, default: 8
            Fontsize used in the plots
        params : dict or None, default: None
            Parameters for matplotlib
        projection : str or None, default: None
           Projection used for the plot
        """
        assert type(orientation) is str
        assert type(fontsize) is int and 0 < fontsize
        assert params is None or type(params) is dict
 
        (width, height) = self._init_orientation(orientation=orientation)
        self._fig = plt.figure(figsize=[width,height], dpi=100, facecolor='w', edgecolor='w')

        #Parameter using subplots
        self._axes = None
        self._nrows = None
        self._ncols = None

        #Parameter for matplotlib
        if params is None:
            params = {'backend': 'pdf',
	              'font.family': 'serif',
                      'font.style': 'normal',
                      'font.variant': 'normal',
                      'font.weight': 'medium',
                      'font.stretch': 'normal',
                      'font.size': fontsize,
                      'font.serif': 'Computer Modern Roman',
                      'font.sans-serif':'Computer Modern Sans serif',
                      'font.cursive':'cursive',
                      'font.fantasy':'fantasy',
                      'font.monospace':'Computer Modern Typewriter',
                      'axes.labelsize': fontsize,
                      'font.size': fontsize, 
                      'legend.fontsize': fontsize,
	              'xtick.major.size': fontsize/3,     
                      'xtick.minor.size': fontsize/4,     
                      'xtick.major.pad': fontsize/4,     
                      'xtick.minor.pad': fontsize/4,     
                      'xtick.color': 'k',     
                      'xtick.labelsize': fontsize,
#                      'xtick.direction': 'in',    
                      'ytick.major.size': fontsize/3,     
                      'ytick.minor.size': fontsize/4,     
                      'ytick.major.pad': fontsize/4,     
                      'ytick.minor.pad': fontsize/4,     
                      'ytick.color': 'k',     
                      'ytick.labelsize': fontsize,
#                      'ytick.direction': 'in',    
	              'savefig.dpi': 320,
	              'savefig.facecolor': 'white',
      	              'savefig.edgecolor': 'white',
                      'lines.linewidth': 0.5,
                      'lines.dashed_pattern': (6, 6),
                      'axes.linewidth': 0.5,
#                      'axes.autolimit_mode': round_numbers,
                      'axes.xmargin': 0.01,
                      'axes.ymargin': 0.02,
    	              'text.usetex': True,
                      'text.latex.preamble': [r'\usepackage{lmodern}', r'\usepackage{siunitx} \DeclareSIUnit[number-unit-product = \,]{\Phosphat}{P} \DeclareSIUnit[number-unit-product = {}]{\Modelyear}{yr} \DeclareSIUnit[number-unit-product = {}]{\Timestep}{dt}', r'\usepackage{xfrac}']}
        
        matplotlib.rcParams.update(params)

        if projection is None:
            self._axesResult = self._fig.add_subplot(111)
        else:
            self._axesResult = self._fig.add_subplot(111, projection=projection)


    def init_subplot(self, nrows, ncols, orientation='lc1', subplot_kw=None, gridspec_kw=None):
        """
        Create a figure with subplots using nrows rows and ncols columns

        Parameters
        ----------
        nrows : int
            Number of rows
        ncols : int
            Number of columns
        orientation : str, default: 'lc1'
            Orientation of the plot
        subplot_kw : dict or None, default: None
            Dict with keywords passed to the add_subplot call used to create
            each subplot
        gridspec_kw : dict or None, default: None
            Dict with keywords passed to the GridSpec construtor used to
            create the grid the subplots are placed on.
        """
        assert type(nrows) is int and 0 < nrows
        assert type(ncols) is int and 0 < ncols
        assert subplot_kw is None or type(subplot_kw) is dict
        assert gridspec_kw is None or type(gridspec_kw) is dict

        self._nrows = nrows
        self._ncols = ncols
        (width, height) = self._init_orientation(orientation=orientation)
        self._fig, self._axes = plt.subplots(nrows=nrows, ncols=ncols, subplot_kw=subplot_kw, gridspec_kw=gridspec_kw, figsize=[width,height], dpi=100, facecolor='w', edgecolor='w')
        
        if nrows == 1 or ncols == 1:
            self._axesResult = self._axes[0]
        else:
            self._axesResult = self._axes[0,0]


    def set_subplot(self, nrow, ncol):
        """
        Set a subplot to plot into

        Parameters
        ----------
        nrow : int
            Number of row
        ncol : int
            Number of column
        @author: Markus Pfeil
        """
        assert self._nrows is not None
        assert self._ncols is not None
        assert type(nrow) is int and 0 <= nrow and nrow < self._nrows
        assert type(ncol) is int and 0 <= ncol and ncol < self._ncols

        if self._nrows == 1:
            self._axesResult = self._axes[ncol]
        elif self._ncols == 1:
            self._axesResult = self._axes[nrow]
        else:
            self._axesResult = self._axes[nrow, ncol]


    def clear_plot(self):
        """
        Clear the current figure
        """
        self._fig.clf()


    def close_fig(self):
        """
        Close the figure windows
        """
        plt.close(self._fig)


    def reinitialize_fig(self, orientation='lc1', fontsize=8):
        """
        Reinitialize the figure

        Close the current figure and generate a new figure

        Parameters
        ----------
        orientation : str, default: 'lc1'
            Orientation of the plot
        fontsize : int, default: 8
            Fontsize used in the plots
        """
        assert type(orientation) is str
        assert type(fontsize) is int and 0 < fontsize

        self.close_fig()
        self._init_plot(orientation=orientation, fontsize=fontsize)


    def set_yscale_log(self, base=10):
        """
        Set y axis to logarithm with given base

        Parameters
        ----------
        base : int or float, default: 10
            Base of the logarithm
        """
        assert type(base) in [int, float] and 0 < base

        self._axesResult.set_yscale('log', basey=base)


    def set_yscale_symlog(self, base=10, linthreshy=10**(-3)):
        """
        Set y axis to symmetric logarithm with given base

        Parameters
        ----------
        base : int or float, default: 10
            Base of the logarithm
        linthreshy : float
            Defines the range(-x, x) within which the plot is linear.
        """
        assert type(base) in [int, float] and 0 < base
        assert type(linthreshy) is float and 0 < linthreshy

        self._axesResult.set_yscale('symlog', basey=base, linthreshy=linthreshy)


    def set_ylim(self, ymin, ymax):
        """
        Set the limit of the y axis

        Parameters
        ----------
        ymin : float
            Minimum value of the y axis
        ymax : float
            Maximum value of the y axis
        """
        assert type(ymin) is float
        assert type(ymax) is float
        assert ymin < ymax

        self.__axesResult.set_ylim([ymin, ymax])


    def set_labels(self, title=None, xlabel=None, xunit=None, ylabel=None, yunit=None):
        """
        Set title and labels of the figure

        Parameters
        ----------
        title : str or None, default: None
            Title of the figure
        xlabel : str or None, default: None
            Label of the x axis
        xunit : str or None, default: None
            Unit of the x axis
        ylabel : str or None, default: None
            Label of the y axis
        yunit : str or None
            Unit of the y axis
        """
        assert title is None or type(title) is str
        assert xlabel is None or type(xlabel) is str
        assert xunit is None or type(xunit) is str
        assert ylabel is None or type(ylabel) is str
        assert yunit is None or type(yunit) is str

        #Set title
        if not title is None:
            self._axesResult.set_title(r'{}'.format(title))
        
        #Set xlabel
        if not xlabel is None:
            if not xunit is None:
                xl = '{} [{}]'.format(xlabel, xunit)
            else:
                xl = xlabel
            self._axesResult.set_xlabel(r'{}'.format(xl))

        #Set ylabel
        if not ylabel is None:
            if not yunit is None:
                yl = '{} [{}]'.format(ylabel, yunit)
            else:
                yl = ylabel
            self._axesResult.set_ylabel(r'{}'.format(yl))


    def set_legend_box(self, bbox_to_anchor=(0,1.02,1,0.2), loc='lower left', mode='expand', borderaxespad=0.0, labelspacing=0.25, borderpad=0.25, ncol=3, handlelength=2.0, handletextpad=0.8):
        """
        Set the legend using a bbox

        Parameters
        ----------
        bbox_to_anchor : 2-tuple or 4-tuple of floats, default: (0,1.02,1,0.2)
            Box that is used to position the legend in conjunction with loc.
        loc : str or int, default: 'lower left'
            Location of the legend
        mode : {'expand', None}, default: 'expand'
            If mode is set to 'expand' the legend will be horizontally
            expanded to fill the axes area.
        borderaxespad : float, default: 0.0
            The pad between the axes and legend border, in font-size units.
        labelspacing : float, default: 0.25
            The vertical space between the legend entries, in font-size units.
        borderpad : float, default: 0.25
            The fractional whitespace inside the legend border, in font-size
            units.
        ncol : int, default: 3
            The number of columns that the legend has.
        handlelength : float, default: 2.0
            The length of the legend handles, in font-size units.
        handletextpad : float, default: 0.8
            The pad between the legend handle and text, in font-size units.
        """
        assert type(bbox_to_anchor) is tuple
        assert type(loc) in [str, int]
        assert mode is None or mode == 'expand'
        assert borderaxespad is None or type(borderaxespad) is float and 0.0 <= borderaxespad
        assert labelspacing is None or type(labelspacing) is float and 0.0 <= labelspacing
        assert borderpad is None or type(borderpad) is float and 0.0 <= borderpad
        assert type(ncol) is int and 0 < ncol
        assert type(handlelength) is float and 0.0 <= handlelength
        assert type(handletextpad) is float and 0.0 <= handletextpad

        self._axesResult.legend(bbox_to_anchor=bbox_to_anchor, loc=loc, mode=mode, borderaxespad=borderaxespad, labelspacing=labelspacing, borderpad=borderpad, ncol=ncol, handlelength=handlelength, handletextpad=handletextpad)


    def set_subplot_adjust(self, left=None, bottom=None, right=None, top=None):
        """
        Adjust the subplot layout parameters.

        Unset parameters are left unmodified.

        Parameters
        ----------
        left : float, optional
            The position of the left edge of the subplots, as a fraction of
            the figure width.
        bottom : float, optional
            The position of the bottom edge of the subplots, as a fraction of
            the figure height.
        right : float, optional
            The position of the right edge of the subplots, as a fraction of
            the figure width.
        top : float, optional
            The position of the top edge of the subplots, as a fraction of the
            figure height.
        """
        assert type(left) is float and 0.0 <= left and left <= 1.0
        assert type(bottom) is float and 0.0 <= bottom and bottom <= 1.0
        assert type(right) is float and 0.0 <= right and right <= 1.0
        assert type(top) is float and 0.0 <= top and top <= 1.0

        self._fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top)


    def savefig(self, filename):
        """
        Save the current figure

        Parameters
        ----------
        filename : str
            Filename of the figure

        Notes
        -----
            If the filename is a path and the directory does not exist, the
            directory is created.
        """
        assert type(filename) is str
        assert type(format) is str

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self._fig.savefig(filename)


    def _reorganize_data(self, v1d):
        """
        Reorganize the tracer concentration vector from 1D to 3D

        Reorganize the tracer concentration vector (v1d) from a one
        dimensional vector to a three dimensional vector using the land sea
        mask.

        Parameters
        ----------
        v1d : numpy.ndarry
            Tracer concentration vector

        Returns
        -------
        numpy.ndarray
            Three dimensional vector of the tracer concentration
        """
        assert type(v1d) is np.ndarray and np.shape(v1d) == (Metos3d_Constants.METOS3D_VECTOR_LEN,)

        z = petsc.readPetscFile(os.path.join(Metos3d_Constants.METOS3D_PATH, 'data', 'data', 'TMM', '2.8', 'Forcing', 'DomainCondition', 'z.petsc'))
        dz = petsc.readPetscFile(os.path.join(Metos3d_Constants.METOS3D_PATH, 'data', 'data', 'TMM', '2.8', 'Forcing', 'DomainCondition', 'dz.petsc'))
        landSeaMask = petsc.readPetscMatrix(os.path.join(Metos3d_Constants.METOS3D_PATH, 'data', 'data', 'TMM', '2.8', 'Geometry', 'landSeaMask.petsc'))
        landSeaMask = landSeaMask.astype(int)

        #Dimensions
        nx, ny = landSeaMask.shape
        nz = 15
        #v3d
        v3d = np.zeros(shape=(3, nx, ny, nz), dtype=float)
        v3d[:,:,:,:] = np.nan

        #v1d -> (v3d, z, dz)
        offset = 0
        for ix in range(nx):
            for iy in range(ny):
                length = landSeaMask[ix, iy]
                if not length == 0:
                    v3d[0, ix, iy, 0:length] = v1d[offset:offset+length]
                    v3d[1, ix, iy, 0:length] = z[offset:offset+length]
                    v3d[2, ix, iy, 0:length] = dz[offset:offset+length]
                    offset = offset + length

        return v3d

