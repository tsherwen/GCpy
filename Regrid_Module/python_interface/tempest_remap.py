'''
Apply Tempest offline regridding weights for Lat-Lon <-> Cube-Sphere remapping.
The code seems to be much cleaner with some object-oriented programming.

Linear regridding weights are stored in the object c2l or l2c,
which can be applied to any in-code data arrays  

It might be also used to generate GMAO tilefiles (TBD)

Jiawei Zhuang 2016/12
'''

import numpy as np
from netCDF4 import Dataset

class LLind:
    '''
    example:
    llind = tempest_remap.LLind(np.arange(1,3313),72,46)
    '''
    def __init__(self,ind1D,Nlon,Nlat):
        '''
        ind1D: lat-lon box numbered by 1D indices. 
        e.g. for (Nlon,Nlat)=(72,46), ind1D can take values from 1,2,...,3312(=72*46) 
        then they are converted to traditional 2D indices (ilon,ilat) 
        which can take values from (1,1),(1,2),...,(72,46)
        '''

        # check dimensions 
        if ind1D.ndim != 1: 
            raise ValueError('ind1D must be an one-dimensional array!')
        if max(ind1D) > Nlon*Nlat or min(ind1D) < 1: 
            raise ValueError('the value of ind1D should be in the range [1,Nlon*Nlat]')

        self.ind1D = ind1D
        self.Nbox = np.size(ind1D) # the number of boxes. 
        # Nbox can be much larger than Nlon*Nlat because there are many pairs to remap

        self.Nlon = Nlon
        self.Nlat = Nlat

        ilat, ilon = np.unravel_index(ind1D-1, (Nlat,Nlon))
        # do not use (Nlon,Nlat) to reshape. python's dimension numbering is different from IDL

        self.ilon = ilon+1
        self.ilat = ilat+1
        # np.unravel_index starts from 0 index,but either Tempest or GMAO tilefile starts from 1.

class CSind:
    '''
    example:
    csind = tempest_remap.CSind(np.arange(1,13825),48)
    csind.rearrange('GMAO')
    '''
    def __init__(self,ind1D,Nx):
        '''
        ind1D: cube-sphere box numbered by 1D indices. 
        e.g. for NX=48 (c48), ind1D can take values from 1,2,...,13824(=48*48*6) 
        then they are converted to 3D indices (ix,iy,ipanel) 
        which can take values from (1,1,1),(1,1,2),...,(6,48,48)

        For GMAO/GCHP format, it should be converted to 2D indices (1,1),...,(288,48) (TBD)
        convert to 2D at last because it is easiest to rearrange faces with 3D indices
        '''
        
        # check dimensions 
        self.ind1D = ind1D
        if ind1D.ndim != 1: 
            raise ValueError('ind1D must be an one-dimensional array!')
        if max(ind1D) > Nx*Nx*6 or min(ind1D) < 1: 
            raise ValueError('the value of ind1D should be in the range [1,Nx*Nx*6]')

        self.ind1D = ind1D
        self.Nbox = np.size(ind1D) # the number of boxes. 
        # Nbox can be much larger than Nx*Nx*6 because there are many pairs to remap

        self.Nx = Nx

        ip,iy,ix = np.unravel_index(ind1D-1, (6,Nx,Nx))
        self.ix = ix+1
        self.iy = iy+1
        self.ip = ip+1
        # np.unravel_index starts from 0 index,but either Tempest or GMAO tilefile starts from 1.

    def switch(self,neworder):
        '''
        neworder is a 6-element array
        if neworder = [1,2,3,4,5,6] then nothing changes
        '''
        for n in range(self.Nbox): # go through all boxes, change ip
            self.ip[n] = neworder[ self.ip[n]-1 ]

    def rotate(self,ipanel,degree):
        '''
        ind_in is a (NX,NX) 2D array
        only rotate the ipanel-th panel 
        '''
        # go through all boxes, only rotate those with a specific ip
        if degree == '90':
            for n in range(self.Nbox): 
                if self.ip[n] in ipanel:
                    self.ix[n],self.iy[n] = self.Nx+1-self.iy[n],self.ix[n] 

        if degree == '180':
            for n in range(self.Nbox): 
                if self.ip[n] in ipanel:
                    self.ix[n] = self.Nx+1-self.ix[n]
                    self.iy[n] = self.Nx+1-self.iy[n]

    def rearrange(self,method):
        '''
        re-arrange indices (switch and rotate panels) to match FV3/GMAO.
        By using a combination of switches and rotations, 
        it should be able to match any kinds of cube-sphere numbering 
        '''

        if method == 'GMAO':
            print('change CS grid box numbering to match GMAO')

            # rotating and switching operators are not inter-changeable!

            # first perform the panel switch
            self.switch([4,5,1,2,6,3])

            # then rotate some panels
            self.rotate([3,4,5],'90')
            self.rotate([6],'180')

        if method == 'other_types_if_necessary':
            pass

class C2L(object):
    '''
    example:
    c2l = tempest_remap.C2L('c48-to-lon72_lat46_MAP_DEPE.nc',72,46,48)
    LLdata = c2l.regrid(CSdata)
    '''
    def __init__(self,infile,Nlon,Nlat,Nx):
        '''
        Open a tempest file and read its regridding weights
        '''
        self.Nlon = Nlon
        self.Nlat = Nlat
        self.Nx   = Nx

        # open the tempest offline regridding file
        print('opening tempest file: '+infile)
        fh = Dataset(infile, "r", format="NETCDF4")

        # check if dimensions are correctly specified
        # reading l2c file for c2l regridding can also cause the error below
        if Nx*Nx*6 != fh.dimensions['n_a'].size:
            raise ValueError('Nx*Nx*6 is not equal to n_a in the file')
        if Nlon*Nlat != fh.dimensions['n_b'].size:
            raise ValueError('Nlon*Nlat is not equal to n_b in the file')
        if Nlon != fh.dimensions['lon_b'].size:
            raise ValueError('Nlon is not equal to lon_b in the file')
        if Nlat != fh.dimensions['lat_b'].size:
            raise ValueError('Nlat is not equal to lat_b in the file')

        # read data from the file
        self.n_a = fh.dimensions['n_a'].size # how many boxes in source grid
        self.n_b = fh.dimensions['n_b'].size # how many boxes in destination grid
        self.n_s = fh.dimensions['n_s'].size # how many pairs of overlapping-boxes

        self.csind = CSind(fh.variables['col'][:], Nx) # box indices of the source grid
        self.llind = LLind(fh.variables['row'][:], Nlon, Nlat) # box indices of the destination grid
        self.csind.rearrange('GMAO') # !!correct the box numbering!!
        self.S = fh.variables['S'][:] # regridding weight
        
        # close netcdf file
        fh.close()
      
        # print basic information
        print('input grid: '+'c'+str(Nx)+
              ' ; total boxes: '+str(self.n_a))
        print('output grid: '+'lon'+str(Nlon)+'_lat'+str(Nlat)+
              ' ; total boxes: '+str(self.n_b))
       
    def regrid(self,indata):
        '''
        regrid indata to outdata online
        '''
        # check input data dimension
        if np.shape(indata) != (6,self.Nx,self.Nx): 
            raise ValueError('input data should have the size (6,Nx,Nx)')

        # array for output data
        outdata = np.zeros([self.Nlat,self.Nlon])

        # explicitly apply regridding weight
        for n in range(self.n_s): # loop over all pairs
            #indices start from 1 but numpy arrays start from 0
            ilon = self.llind.ilon[n] - 1
            ilat = self.llind.ilat[n] - 1
            ix   = self.csind.ix[n] - 1
            iy   = self.csind.iy[n] - 1
            ip   = self.csind.ip[n] - 1
            outdata[ilat,ilon] += self.S[n]*indata[ip,iy,ix]

        return outdata

    def tilefile(self,filename):
        '''
        We don't seem to need C2L tilefiles
        '''
        pass


class L2C(object):
    '''
    example:
    l2c = tempest_remap.L2C('lon72_lat46-to-c48_MAP_DEPE.nc',72,46,48)
    CSdata = l2c.regrid(LLdata)
    '''
    def __init__(self,infile,Nlon,Nlat,Nx):
        '''
        Open a tempest file and read its regridding weights
        '''
        self.Nlon = Nlon
        self.Nlat = Nlat
        self.Nx   = Nx

        # open the tempest offline regridding file
        print('opening tempest file: '+infile)
        fh = Dataset(infile, "r", format="NETCDF4")

        # check if dimensions are correctly specified
        # reading c2l file for l2c regridding can also cause the error below
        if Nx*Nx*6 != fh.dimensions['n_b'].size:
            raise ValueError('Nx*Nx*6 is not equal to n_b in the file')
        if Nlon*Nlat != fh.dimensions['n_a'].size:
            raise ValueError('Nlon*Nlat is not equal to n_a in the file')
        # in L2C offline MAP, there is no lat_a and lon_a for checking, so be careful.

        # read data from the file
        self.n_a = fh.dimensions['n_a'].size # how many boxes in source grid
        self.n_b = fh.dimensions['n_b'].size # how many boxes in destination grid
        self.n_s = fh.dimensions['n_s'].size # how many pairs of overlapping-boxes

        self.llind = LLind(fh.variables['col'][:], Nlon, Nlat) # box indices of the source grid
        self.csind = CSind(fh.variables['row'][:], Nx) # box indices of the destination grid
        self.csind.rearrange('GMAO') # !!correct the box numbering!!
        self.S = fh.variables['S'][:] # regridding weight
        
        # close netcdf file
        fh.close()
      
        # print basic information
        print('input grid: '+'lon'+str(Nlon)+'_lat'+str(Nlat)+
              ' ; total boxes: '+str(self.n_a))
        print('output grid: '+'c'+str(Nx)+
              ' ; total boxes: '+str(self.n_b))
       
    def regrid(self,indata):
        '''
        regrid indata to outdata online
        '''
        # check input data dimension
        if np.shape(indata) != (self.Nlat,self.Nlon): 
            raise ValueError('input data should have the size (Nlat,Nlon)')

        # array for output data
        outdata = np.zeros([6,self.Nx,self.Nx])

        # explicitly apply regridding weight
        for n in range(self.n_s): # loop over all pairs
            #indices start from 1 but numpy arrays start from 0
            ilon = self.llind.ilon[n] - 1
            ilat = self.llind.ilat[n] - 1
            ix   = self.csind.ix[n] - 1
            iy   = self.csind.iy[n] - 1
            ip   = self.csind.ip[n] - 1
            outdata[ip,iy,ix] += self.S[n]*indata[ilat,ilon]

        return outdata

    def tilefile(self,filename):
        '''
        Create a GMAO tilefile (TBD)
        A tilefile needs:
        self.llind.ilon
        self.llind.ilat
        self.csind.ix
        self.csind.iy_2D ( = iy+Nx*(ip-1)  )
        self.S
        All of them are already known once the object is initialized
        The only thing left is binary file I/O
        '''
        iy_2d = self.csind.iy + self.Nx*(self.csind.ip-1)
        for n in range(self.n_s):
            print(iy_2d[n],self.csind.iy[n],self.csind.ip[n])
        print(np.shape(iy_2d))
