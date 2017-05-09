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
        # We have to make use of numpy vectorization to process
        # 3D data that contains many levels, otherwise the regridding
        # will be deadly slow.
        # Can be extended to 4D data (contains time) if necessary.
        ndim = np.ndim(indata)
        sp = np.shape(indata)
        if ndim == 2:
            if sp == (6*self.Nx, self.Nx):
                # [6*Nx,6]: 2D field with 6 panels squeezed to y-axis
                indata = np.reshape(indata,[6,self.Nx,self.Nx]) # separate the panel axis
                ndim = 3 # now it is a 3D array
                outdata = np.zeros([self.Nlat,self.Nlon])
            else:
                raise ValueError('2D input array should have the size (6*Nx,Nx)')
        elif ndim == 3:
            if sp == (6, self.Nx, self.Nx): 
                # [6,Nx,NX]: 2D field with 6 panels being a separate dimension 
                outdata = np.zeros([self.Nlat,self.Nlon])
            elif (sp[1],sp[2]) == (6*self.Nx, self.Nx):
                # [Nlev,6*Nx,Nx]: 3D field with 6 panels squeezed to y-axis

                # separate the panel axis
                # only use np.reshape will mess up the axes. so we do two steps
                # [Nlev,6*Nx,Nx] -> [Nlev,6,Nx,Nx] (we have to reshape adjacent dimensions first)
                indata = np.reshape(indata,[sp[0],6,self.Nx,self.Nx])
                # [Nlev,6,Nx,Nx] -> [6,Nlev,Nx,Nx] (then swap dimensions)
                indata = np.swapaxes(indata,0,1)

                ndim = 4 # now it is a 4D array
                outdata = np.zeros([sp[0],self.Nlat,self.Nlon])
            else:
                raise ValueError('3D input array should have the size of'+ 
                        '[6,Nx,Nx](2D field) or [Nlev,6*Nx,Nx](3d field)')
        elif ndim == 4:
            if (sp[0],sp[2],sp[3]) == (6,self.Nx,self.Nx):
                # [6,Nlev,Nx,Nx]: 3D field with 6 panels being a separate dimension
                outdata = np.zeros([sp[1],self.Nlat,self.Nlon])
            else:
                raise ValueError('4D input array should have the size [6,Nlev,Nx,Nx](3d field)')
        else:
            raise ValueError('invalid dimension of input data')

        # explicitly apply regridding weight
        for n in range(self.n_s): # loop over all pairs
            #indices start from 1 but numpy arrays start from 0
            ilon = self.llind.ilon[n] - 1
            ilat = self.llind.ilat[n] - 1
            ix   = self.csind.ix[n] - 1
            iy   = self.csind.iy[n] - 1
            ip   = self.csind.ip[n] - 1

            if ndim == 3:
                # Tons of unnecessary branching, but doesn't seem to affect performance.
                # Moving this if clause outside of the loop will introduce repeated code.
                outdata[ilat,ilon] += self.S[n]*indata[ip,iy,ix]
            elif ndim == 4:
                outdata[:,ilat,ilon] += self.S[n]*indata[ip,:,iy,ix]

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
        Create a GMAO-style tilefile

        Define blocks of data to write, each consisting of data
        memory (bytes), data, and byte format string. Pack each
        block into a python bytes object for writing. Number
        of bytes is written twice for read error checking.

        NOTES:
           1. Byte format str mapping is : i=int32, p=char, f=float32         
           2. dummy variables are not used but are needed to meet
              format requirements
        '''
        dmem = self.n_s*4                   # num bytes per map array
        # pad grid names to be 128 bytes
        llname = self.llind.name.ljust(128,' ')
        csname = self.csind.name.ljust(128,' ')
        # Get the 2D indexes
        jj_ii_2d = self.csind.get_2D_ind()
        data_blocks = [
            ( 4,    self.n_s,          'i' ), # num overlapping boxes
            ( 4,    2,                 'i' ), # num grids
            ( 128,  llname,            'p' ), # 128-byte lat/lon grid name 
            ( 4,    self.llind.Nlon,   'i' ), # lat/lon num lons
            ( 4,    self.llind.Nlat,   'i' ), # lat/lon num lats
            ( 128,  csname,            'p' ), # 128-byte cs grid name
            ( 4,    self.csind.NX,     'i' ), # cs face side length
            ( 4,    self.csind.NX*6,   'i' ), # face side length * 6
            ( dmem, [-1]*self.n_s,     'f' ), # dummy type variable
            ( dmem, [-1]*self.n_s,     'f' ), # dummy x variable
            ( dmem, [-1]*self.n_s,     'f' ), # dummy y variable
            ( dmem, self.llind.ilon+1, 'f' ), # 1-based lon mapping indexes
            ( dmem, self.llind.ilat+1, 'f' ), # 1-based lat mapping indexes
            ( dmem, self.S,            'f' ), # regridding weights
            ( dmem, self.csind.ii+1,   'f' ), # 1-based cs equivalent of ilon
            ( dmem, self.csind.jj+1,   'f' ), # 1-based cs equivalent of ilat
            ( dmem, self.S,            'f' )  # regridding weights (maybe change to bad vals since not used)
        ]
        with open( filename, 'wb') as f: 
            for dbytes, dname, dtype in data_blocks:
                s = struct.Struct('i {} i'.format(dtype))
                if dtype == 'p':
                    dname = str.encode(dname)
                f.write(s.pack(dbytes, dname, dbytes)) 

class L2L(object):
    '''
    example:
    l2l = tempest_remap.L2L('72x46_DEPC-to-288x181_DEPC.nc',72,46,288,181)
    CSdata = l2c.regrid(LLdata)
    '''
    def __init__(self,infile,Nlon_in,Nlat_in,Nlon_out,Nlat_out):
        '''
        Open a tempest file and read its regridding weights
        '''
        self.Nlon_in = Nlon_in
        self.Nlat_in = Nlat_in
        self.Nlon_out = Nlon_out
        self.Nlat_out = Nlat_out

        # open the tempest offline regridding file
        print('opening tempest file: '+infile)
        fh = Dataset(infile, "r", format="NETCDF4")

        # check if dimensions are correctly specified
        if Nlon_in*Nlat_in != fh.dimensions['n_a'].size:
            raise ValueError('Nlon_in*Nlat_in is not equal to n_a in the file')
        if Nlon_out*Nlat_out != fh.dimensions['n_b'].size:
            raise ValueError('Nlon_out*Nlat_out is not equal to n_b in the file')

        # read data from the file
        self.n_a = fh.dimensions['n_a'].size # how many boxes in source grid
        self.n_b = fh.dimensions['n_b'].size # how many boxes in destination grid
        self.n_s = fh.dimensions['n_s'].size # how many pairs of overlapping-boxes

        self.llind_in = LLind(fh.variables['col'][:], Nlon_in, Nlat_in) # box indices of the source grid
        self.llind_out = LLind(fh.variables['row'][:], Nlon_out, Nlat_out) # box indices of the destination grid
        self.S = fh.variables['S'][:] # regridding weight
        
        # close netcdf file
        fh.close()
      
        # print basic information
        print('input grid: '+'lon'+str(Nlon_in)+'_lat'+str(Nlat_in)+
              ' ; total boxes: '+str(self.n_a))
        print('input grid: '+'lon'+str(Nlon_out)+'_lat'+str(Nlat_out)+
              ' ; total boxes: '+str(self.n_b))
       
    def regrid(self,indata):
        '''
        regrid indata to outdata online
        '''
        # check input data dimension
        ndim = np.ndim(indata)
        sp = np.shape(indata)
        if ndim == 2:
            if sp != (self.Nlat_in,self.Nlon_in): 
                 raise ValueError('2D input data should have the size (Nlat,Nlon)') 
            outdata = np.zeros([self.Nlat_out,self.Nlon_out])
        elif ndim == 3:
            if (sp[1],sp[2]) != (self.Nlat_in,self.Nlon_in): 
                raise ValueError('3D input data should have the size (Nlev,Nlat,Nlon)')
            outdata = np.zeros([sp[0],self.Nlat_out,self.Nlon_out])
        else:
            raise ValueError('invalid dimension of input data')

        # array for output data

        # explicitly apply regridding weight
        for n in range(self.n_s): # loop over all pairs
            #indices start from 1 but numpy arrays start from 0
            ilon_in  = self.llind_in.ilon[n] - 1
            ilat_in  = self.llind_in.ilat[n] - 1
            ilon_out = self.llind_out.ilon[n] - 1
            ilat_out = self.llind_out.ilat[n] - 1
            if ndim == 2:
                outdata[ilat_out,ilon_out] += self.S[n]*indata[ilat_in,ilon_in]
            elif ndim == 3:
                outdata[:,ilat_out,ilon_out] += self.S[n]*indata[:,ilat_in,ilon_in]

        return outdata

