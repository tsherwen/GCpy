import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import os
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import ListedColormap

# get gamap's WhGrYlRd color scheme from file
current_dir = os.path.dirname(os.path.abspath(__file__))
WhGrYlRd_scheme = np.genfromtxt(current_dir+'/colormap/WhGrYlRd.txt',delimiter=' ')
WhGrYlRd = ListedColormap(WhGrYlRd_scheme/255.0)

class CSlayer(object):

    def __init__(self,Nx,lon_shift=0.0):

        # looking for the grid information
        self.Nx = Nx
        self.grid_lon = []
        self.grid_lat = []
        
        griddir = current_dir+"/CSgridinfo/C{0}/".format(Nx)
        print('search gridinfo in the directory: \n',griddir)
        for ip in range(1,7):  
             filename = griddir+"atmos_static.tile{0}.nc".format(ip)
             fh = Dataset(filename)

             # without copy() we'll get the address instead of the static value
             self.grid_lon.append(fh["grid_lon"][:].copy()) 
             self.grid_lat.append(fh["grid_lat"][:].copy())
             fh.close()

        # no matter shift or not, we need to correct the range because FV3 outputs in [0,360]
        for ip in range(6):
            grid_lon = self.grid_lon[ip] #just to make equations shorter
            grid_lon += lon_shift
        
            # keep the range in [-180~180]
            grid_lon[grid_lon > 180] -= 360
            grid_lon[grid_lon < -180] += 360
        
            # we ARE modifying self.grid_lon[ip],
            # because grid_lon share the same address with it.
            # no need to write back.
            del grid_lon


        # mask the boxes at the boundary of the figure
        # pcolormesh cannot handle boxes across the boundary
        max_stride = 180 # the maximum distance of the 4 corners of a cell
        self.mask = []
        for ip in range(6):
            maski = np.abs(np.diff(self.grid_lon[ip],axis=0)) > max_stride
            maskj = np.abs(np.diff(self.grid_lon[ip],axis=1)) > max_stride
            self.mask.append( maski[:,1:] | maski[:,:-1] | maskj[1:,:] | maskj[:-1,:] )
        
    def tvmap(self,indata,axis=None,vmax=None,vmin=None,
              title='',unit='unit',continent=True, grid=True,ticks=True,
              cmap=WhGrYlRd):

        if isinstance(indata,list):
            # A list of 6 elements, each contains a [Nx,Nx] numpy array
            # often seen in online FV3 output (separate tile files) 
            if len(indata) == 6:
                data = indata # we want a list
            else:
                raise ValueError('the length of input list should be 6')

        elif isinstance(indata,np.ndarray):
            # A single numpy array. 
            # often seen in GCHP output (a single file)
            if np.shape(indata) == (6*self.Nx,self.Nx):
                indata = np.reshape(indata,[6,self.Nx,self.Nx])
            elif np.shape(indata) == (6,self.Nx,self.Nx):
                pass
            else:
                raise ValueError('the shape of input array should be [6*Nx,NX] or [6,Nx,Nx]')
        
            data = [] # we want a list
            for ip in range(6):
                    data.append(indata[ip,:,:])
            del indata # don't need indata anymore

        # mask the boxes at the boundary of the figure
        for ip in range(6): 
            data[ip] = np.ma.masked_where(self.mask[ip],data[ip])

        # If axis is not specified, create a new one. Otherwise plot on the given axis
        show = False
        if axis is None:
             plt.figure(figsize=(12,6))
             axis = plt.gca()
             # If axis is not specified, show the plot immediately 
             show = True      
  
        m = Basemap(ax=axis,projection='cyl',lon_0=0,fix_aspect=False) 
        #lon_0 is the center longitude
        
        if continent: m.drawcoastlines()
        if grid :
            if ticks:
                labels=[1,0,0,0]
            else:
                labels=[0,0,0,0]
            m.drawparallels(np.arange(-90,91,30),labels=labels)
            if ticks:
                labels=[0,0,0,1]
            else:
                labels=[0,0,0,0]
            m.drawmeridians(np.arange(-180,180,60),labels=labels)
        
        # must use the same color scale for all panels!
        if vmax is None:
            vmax = np.max(data) 
        if vmin is None:
            vmin = np.min(data) 

        for ip in range(6):
            # There's a bug that with latlon=True, masks don't work correctly
            im = m.pcolormesh(self.grid_lon[ip],self.grid_lat[ip],data[ip],
                              latlon=False,vmax=vmax,vmin=vmin,cmap=cmap,
                              linewidth=0.0,rasterized=True)
       
        cb = m.colorbar(im)
        cb.ax.set_title(unit,fontsize=12,y=1.0,x=2.0) # on the top of the color bar
        plt.title(title)

        if show: plt.show() 
