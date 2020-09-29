#!/usr/bin/env python
# -*- coding: utf8 -*

import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Plot():

    def __init__(self, cmap=None, orientation='lc1', fontsize=8, params=None, projection=None):
        """
        Initialize the plot
        @author: Markus Pfeil
        """
        assert type(orientation) is str
        assert type(fontsize) is int and 0 < fontsize

        #Colors
        self._colors = {
                -1: 'black', 
                 0: 'C5', 
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
        Initialize the orientation and size of the plot
        @author: Markus Pfeil
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
        Initialize the plot windows.
        The title and axes labels are set.
        @author: Markus Pfeil
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
                      'text.latex.preamble': [r'\usepackage{lmodern}', r'\usepackage{siunitx} \DeclareSIUnit[number-unit-product = \,] \Phosphat{P} \DeclareSIUnit[number-unit-product = {}] \Modelyear{yr} \DeclareSIUnit[number-unit-product = {}] \Timestep{dt}', r'\usepackage{xfrac}']}
        
        matplotlib.rcParams.update(params)

        if projection is None:
            self._axesResult = self._fig.add_subplot(111)
        else:
            self._axesResult = self._fig.add_subplot(111, projection=projection)


    def init_subplot(self, nrows, ncols, orientation='lc1', subplot_kw=None, gridspec_kw=None):
        """
        Create a figure with subplots using nrows rows and ncols columns.
        @author: Markus Pfeil
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
        Set a subplot to plot into.
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
        @author: Markus Pfeil
        """
        self._fig.clf()


    def close_fig(self):
        """
        Close the figure.
        @author: Markus Pfeil
        """
        plt.close(self._fig)


    def reinitialize_fig(self, orientation='lc1', fontsize=8):
        """
        Reinitialize the figure
        Close the current figure and generate a new figure.
        @author: Markus Pfeil
        """
        assert type(orientation) is str
        assert type(fontsize) is int and 0 < fontsize

        self.close_fig()
        self._init_plot(orientation=orientation, fontsize=fontsize)


    def set_yscale_log(self, base=10):
        """
        Set y axis to logarithm with given base.
        @author: Markus Pfeil
        """
        assert type(base) in [int, float] and 0 < base

        self._axesResult.set_yscale('log', basey=base)


    def set_labels(self, title=None, xlabel=None, xunit=None, ylabel=None, yunit=None):
        """
        Set title and labels of the figure.
        @author: Markus Pfeil
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


    def set_subplot_adjust(self, left=None, bottom=None, right=None, top=None):
        """
        Set Subplot adjust of the figure.
        @author: Markus Pfeil
        """
        assert type(left) is float and 0.0 <= left and left <= 1.0
        assert type(bottom) is float and 0.0 <= bottom and bottom <= 1.0
        assert type(right) is float and 0.0 <= right and right <= 1.0
        assert type(top) is float and 0.0 <= top and top <= 1.0

        self._fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top)


    def savefig(self, filename, format='pdf'):
        """
        Save figure
        @author: Markus Pfeil
        """
        self._fig.savefig(filename)
