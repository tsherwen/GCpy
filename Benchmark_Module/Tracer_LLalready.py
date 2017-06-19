"""
PURPOSE:
    This module assumes all data are already on the same Lat-Lon Grid
    
    GEOSChem/GCHP python benchmark tool. 

NOTES:
    
    1) Assume python3 syntax. Not tested extensively with python2.
    
    2) The key functionality is comparing two models, no matter
       (GCC,GCHP), (GCC,GCC), or (GCHP,GCHP).
       
    3) Currently only benchmark instantaneous tracer mixing ratio.
    
    4) For GEOSChem-classic, only support netCDF files. Will NOT make any
       effort to support bpch format, because NC diagnostics are already  
       implemented in v11-01, and should become standard in v11-02
    
    5) Only use the most basic packages (netCDF4,matplotlib,Basemap) 
       to gain full control over the plot style and format. 

REVISION HISTORY:
    12 Feb 2017 - J.W.Zhuang  - Initial version
    17 Apr 2017 - E. Lundgren - Updates for v1.0.0 benchmark
    See git history for history of all further revisions

"""

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from matplotlib.backends.backend_pdf import PdfPages
from itertools import chain

from Visualize_Module import gamap as ga

class benchmark:
    '''
    Compare two GEOS-Chem NetCDF outputs (either classic or HP), including 
    reading data, making standard plots and quantifying error. 
    '''
    
    def __init__(self,outputdir='./',shortname='DefaultName',longname=None):
        '''
        Just a routinely method to initialize the "benchmark" object.
        Not doing anything meaningful except for setting the case name. 
        [ can be viewed as something like plt.figure() ]
        The most important parameters are further initialized by the 
        getdata1 method.
        
        Parameters
        ----------
        outputdir: string, optional but recommended
            To output directory plots. Use the current dir by default.
            It should end with '/'
            
        shortname: string, optional but recommended
            The shortname of this test. 
            Will be used as the file names of output plots
            
        longname: string, optional but recommended
            The longname of this test. 
            Will be used as the titles of output plots

        '''
        self.outputdir=outputdir
        self.shortname=shortname
        if isinstance(longname, type(None)):
            self.longname=shortname
        else:
            self.longname=longname
            
        self.isData1=False # haven't read data1
        
    def getdata1(self,filename,tracerlist=None,tag='model1',
                 prefix='IJ_AVG_S__',time=0,flip=False,avg_time=False):
        '''
        Get data from the 1st model.
        
        Parameters
        ----------
        filename: string
            The name of the netCDF output file
    
        tracerlist: list of strings, optional
            The names of tracers to be extracted from the netCDF file.
            If not specified, then try to extract all the tracers that 
            matches the prefix.
            
        tag: string, optional but recommended
            The name of the model shown in the plots
            
        prefix: string, optional
            The prefix of variable names in the netCDF file
            
        time: integer, optional
            the time slice to extract. get the first slice by default
        
        flip: logic, optional
            flip the lev dimension or not. Mainly for GCHP I/O issue.
        
        
        Important Returns
        ----------
        self.tracername: list of strings
            tracer names without prefix
        
        self.data1: list of 3D numpy arrays
            3D (lev,lat,lon) data at one time slice.
            
        '''
        
        # open the netcdf file with read-only mode
        fh = Dataset(filename, "r", format="NETCDF4")

        # Temporally set override coridinate variable names
#        nc_lon_var = 'lon'
#        nc_lat_var = 'lat'
#        nc_lev_var = 'lev'
        nc_lon_var = 'longitude'
        nc_lat_var = 'latitude'
        nc_lev_var = 'model_level_number'#'dim3'
        print( [ i for i in fh.variables ] )
        # get dimension info
        # use len() instead of .size for compatibility with python2
        self.lon=fh[nc_lon_var][:]
        self.lat=fh[nc_lat_var][:]
        self.lev=fh[nc_lev_var][:]
        self.Nlon=len(self.lon)
        self.Nlat=len(self.lat)
        self.Nlev=len(self.lev)
        
        # initialize an empty list to hold the data.
        self.data1=[]

        # initialize an empty dictionary to hold the units
        self.data1_tracer_units={}     
        
        # use 'is' instead of '==' to test None
        if isinstance(tracerlist, type(None)):
            # tracerlist not specified. go through all entries
            
            # initialize an empty list to hold the name
            self.tracername=[]
            
            # get the variable information. fh.variables is a dictionary 
            # consisting of key-value pairs. The key is the variable name 
            # and the value is all the other information
            var = fh.variables
            
            # the standard way to go through a dictionary
            # loop over (key,value) pairs 
            for k,v in var.items():
                # skip the dimension variables
                if prefix in k:
                    # get the name without prefix
                    self.tracername.append(k.replace(prefix,''))
                    # average time of get one time slice?
                    if avg_time:
                        data=v[:].mean(axis=0)
                    else:
                        data=v[time,:,:,:]
                    # flip lev dimension of data?
                    if flip: data=data[::-1,:,:]
                    self.data1.append(data)
                    # save the units of the data 
                    self.data1_tracer_units[varname] = fh[varname].ctm_units            
        else:
            # tracerlist specified. 
            self.tracername=tracerlist
            
            for tracer in tracerlist:
                # expand, for example, 'O3' to 'SPC_O3'
                varname=prefix+tracer
                # extract the data directly by the key
                data=fh[varname][:]
                # average time of get one time slice?
                if avg_time:
                    data=data.mean(axis=0)
                else:
                    data=data[time,:,:,:]    
                # flip lev dimension of data?
                if flip: data=data[::-1,:,:]
                self.data1.append(data)
                # save the units of the data 
                self.data1_tracer_units[tracer] = fh[varname].ctm_units

        # always remember to close the NC file
        fh.close() 
        
        # data1 is read
        self.isData1=True
        
        self.model1=tag # for plot names
        
    def getdata2(self,filename,tag='model2',
                 prefix='IJ_AVG_S__',time=0,flip=False,avg_time=False):
        '''
        Get data from the 2nd model, with the tracerlist got from getdata1 
        Must run getdata1 first. data2 will match data1's tracer sequence. 
        
        Parameters
        ----------
        see getdata1. The only difference is not requiring tracerlist input
        
        Impoertant Returns
        ----------        
        self.data2: list of 3D numpy arrays
            3D (lev,lat,lon) data at one time slice.
        
        '''
        if self.isData1 == False:
            raise ValueError('must run getdata1 first')
            
        # open the netcdf file with read-only mode
        fh = Dataset(filename, "r", format="NETCDF4")

        # Temporally set override coridinate variable names
#        nc_lon_var = 'lon'
#        nc_lat_var = 'lat'
#        nc_lev_var = 'lev'
        nc_lon_var = 'longitude'
        nc_lat_var = 'latitude'
        nc_lev_var = 'model_level_number'#'dim3'
        print( [ i for i in fh.variables ] )
        # get dimension info
        # use len() instead of .size for compatibility with python2

        # check dimension 
        if self.Nlon != len(fh[nc_lon_var][:]):
            raise ValueError('lon dimension does not match')
        if self.Nlat != len(fh[nc_lat_var][:]):
            raise ValueError('lat dimension does not match')
#        if self.Nlev != len(fh[nc_lev_var][:]):
#            raise ValueError('lev dimension does not match')
        
        # initialize an empty list to hold the data.
        self.data2=[]

        # initialize an empty dictionary to hold the units
        self.data2_tracer_units={}        
            
        for tracer in self.tracername:
            # expand, for example, 'O3' to 'SPC_O3'
            varname=prefix+tracer
            # extract the data directly by the key
            data=fh[varname][:]
            # average time of get one time slice?
            if avg_time:
                data=data.mean(axis=0)
            else:
                data=data[time,:,:,:] 
            # flip lev dimension of data?
            if flip: data=data[::-1,:,:]
            self.data2.append(data)
            # save the units of the data 
            self.data2_tracer_units[tracer] = fh[varname].ctm_units
            
        # always remember to close the NC file
        fh.close() 
        
        self.model2=tag # for plot names

#     def convert_NetCDF_shape(self, arr):
#         """
#         Convert PyGChem(iris) array arrangement to COARDS
#         
#         PyGChem(iris) saves NetCDF files with 
# 
#         e.g. 
#         (time, longitude, latitude, model_level_number)
#         instead of 
#         (time,lev,lat,lon)
#         of Fortran read
#         (lon,lat,lev,time)
#         
#         """
        
    def getdata0(self,filename,
                 prefix='SPC_',time=0,flip=False):
        '''
        Get initial condition, with the tracerlist got from getdata1 
        Must run getdata1 first. data0 will match data1's tracer sequence. 
        
        Parameters
        ----------
        see getdata1. The only difference is not requiring tracerlist input
        
        Impoertant Returns
        ----------        
        self.data0: list of 3D numpy arrays
            3D (lev,lat,lon) data at one time slice.
        
        '''
        if self.isData1 == False:
            raise ValueError('must run getdata1 first')
            
        # open the netcdf file with read-only mode
        fh = Dataset(filename, "r", format="NETCDF4")
            
        # check dimension 
        if self.Nlon != len(fh['lon'][:]):
            raise ValueError('lon dimension does not match')
        if self.Nlat != len(fh['lat'][:]):
            raise ValueError('lat dimension does not match')
        if self.Nlev != len(fh['lev'][:]):
            raise ValueError('lev dimension does not match')

        # initialize an empty list to hold the data.
        self.data0=[]
            
        for tracer in self.tracername:
            # expand, for example, 'O3' to 'SPC_O3'
            varname=prefix+tracer
            # extract the data directly by the key
            data=fh[varname][time,:,:,:]
    
            if flip: data=data[::-1,:,:]
            self.data0.append(data)
            
        # always remember to close the NC file
        fh.close() 
      
    def plot_layer(self,lev=0,plot_change=False,switch_scale=False,
                   pdfname='default_layerplot.pdf',tag='',
                   rows=3):
        '''
        Compare self.data1 and self.data2 on a specific level. Loop over all
        tracers in self.tracerlist. Plot on one multi-page pdf file. 
        
        Parameters
        ----------
        lev: integer
            The level to compare
            
        pdfname: string, optional but recommended
            The name the output pdf file
            
        tag: string, optional but recommended
            Will be shown as part of the title. For example,'surface', '500hpa'
            
        rows: integer, optional
            How many rows on a page. Although the page size will be 
            adjusted accordingly, 3 rows look better.
            
        plot_change: logical, optional
            If set to True, then plot the change with respect to the initial
            condition (i.e. self.data0), rather than the original field.
            In this case, self.getdata0 must be executed before.
        
        switch_scale: logical, optional
            By default, the scale of data2 is used for both data1 and data2.
            This is because model2 is often regarded as the more correct one
            (the one for reference), so if model1 goes crazy, the color scale 
            will still make sense.
            
            If set to True, then use the scale of data1 instead. This is often
            combined with plot_change=True to more clearly show the regridding
            error.
            
    
        Important Returns
        ----------
            A pdf file containing the plots of all tracers on that level.
        
        '''
        
        N = len(self.tracername) # number of tracers
        Npages = (N-1)//rows+1 # have many pdf pages needed
        
        print('create ',pdfname)
        pdf=PdfPages(self.outputdir+pdfname) # create the pdf to save figures
        
        for ipage in range(Npages):
            
            fig, axarr = plt.subplots(rows,3,figsize=(12,rows*3))
        
            for i in range(rows):
                i_tracer = ipage*rows + i
                print('plotting: no.',i_tracer+1,self.tracername[i_tracer])
                
                # make variable names shorter for simplicity
                tracername = self.tracername[i_tracer]
                
                # assume v/v, convert to ppbv
                # might need modification in the future
                # TMS EDIT - PyGChem/iris save to NetCDF alters aixs order
                # Update from (time, longitude, latitude, model_level_number)
                # to: (time,lev,lat,lon)
#                data1 = self.data1[i_tracer][lev,:,:]#*1e9
#                data2 = self.data2[i_tracer][lev,:,:]#*1e9
                data1 = self.data1[i_tracer][:,:,lev].T#*1e9
                data2 = self.data2[i_tracer][:,:,lev].T#*1e9
#                unit self.data2_tracer_units
                unit = self.data2_tracer_units[tracername]
#                unit='ppbv'
                
                if plot_change:
                    # plot the change with respect to the initial condition,
                    # instead to the original field
                    # Update axis order 
#                    data0 = self.data0[i_tracer][lev,:,:]#*1e9
                    data0 = self.data0[i_tracer][:,:,lev].T#*1e9
                    data1 -= data0
                    data2 -= data0
                    cmap=plt.cm.RdBu_r
                    
                else:
                    cmap=ga.WhGrYlRd
                
                if np.max(np.abs(data1)) < 1e-1 :
                    if unit=='ppbv':
                        data1 *= 1e3
                        data2 *= 1e3
                        unit='pptv'             
                elif np.max(np.abs(data1)) > 1e3 :
                    if unit=='ppbv':
                        data1 /= 1e3
                        data2 /= 1e3
                        unit='ppmv'
                                  
                # use the same scale for data1 and data2
                # use the scale of data2 by default.
                if switch_scale:
                    range_data = np.max(np.abs(data1))
                else:
                    range_data = np.max(np.abs(data2))
                # calculate the difference between two data sets
                data_diff = data1-data2
                range_diff=np.max(np.abs(data_diff))
                
                if plot_change:
                    vmin = -range_data
                    vmax = range_data
                    title= tracername+' change: '
                else:
                    vmin = 0
                    vmax = range_data
                    title= tracername+': '

                # overwrite much of above
                data1_min = np.min(np.abs(data1))
                data1_max = np.max(np.abs(data1))
                data2_min = np.min(np.abs(data2))
                data2_max = np.max(np.abs(data2))
                data_min =  np.min([data1_min,data2_min])
                data_max =  np.max([data1_max,data2_max]) 

                # always use max of both datasets instead
                vmax = data_max

                # use 0 for min unless dataset range is small 
                if (data_max-data_min)/data_max < 0.2:
                    vmin = data_min

                ga.tvmap(data1,axis=axarr[i,0],vmin=vmin,vmax=vmax,unit=unit,
                         cmap=cmap,title=title+self.model1,ticks = False)
                ga.tvmap(data2,axis=axarr[i,1],vmin=vmin,vmax=vmax,unit=unit,
                         cmap=cmap,title=title+self.model2,ticks = False)
                ga.tvmap(data_diff,axis=axarr[i,2],unit=unit,
                         title=title+self.model1+' — '+self.model2,
                         ticks = False,
                         cmap=plt.cm.RdBu_r,vmax=range_diff,vmin=-range_diff)
                
                if i_tracer+1 == N : 
                    i_hide = rows-1
                    while(i_hide > i):
                        [a.axis('off') for a in axarr[i_hide, :]]
                        i_hide -= 1
                    break # not using the full page
                
            fig.suptitle(self.longname+': '+tag,fontsize=15)

            print('saving one pdf page')
            pdf.savefig(fig)  # saves the figure into a pdf page
            plt.close() # close the current figure, prepare for the next page
            
        pdf.close() # close the pdf after saving all figures
        print(pdfname,' finished')
        
    def plot_fracDiff(self,lev=0,plot_change=False,switch_scale=False,
                   pdfname='default_fracDiffplot.pdf',full_range=False,
                   tag='',rows=3):
        '''
        Compare self.data1 and self.data2 on a specific level. Loop over all
        tracers in self.tracerlist. Plot on one multi-page pdf file. 
        
        Parameters
        ----------
        lev: integer
            The level to compare
            
        pdfname: string, optional but recommended
            The name the output pdf file
            
        tag: string, optional but recommended
            Will be shown as part of the title. For example,'surface', '500hpa'
            
        rows: integer, optional
            How many rows on a page. Although the page size will be 
            adjusted accordingly, 3 rows look better.
            
        plot_change: logical, optional
            If set to True, then plot the change with respect to the initial
            condition (i.e. self.data0), rather than the original field.
            In this case, self.getdata0 must be executed before.
        
        switch_scale: logical, optional
            By default, the scale of data2 is used for both data1 and data2.
            This is because model2 is often regarded as the more correct one
            (the one for reference), so if model1 goes crazy, the color scale 
            will still make sense.
            
            If set to True, then use the scale of data1 instead. This is often
            combined with plot_change=True to more clearly show the regridding
            error.
            
    
        Important Returns
        ----------
            A pdf file containing the plots of all tracers on that level.
        
        '''
        
        N = len(self.tracername) # number of tracers
        Npages = (N-1)//rows+1 # have many pdf pages needed
        
        print('create ',pdfname)
        pdf=PdfPages(self.outputdir+pdfname) # create the pdf to save figures
        
        for ipage in range(Npages):
            
            fig, axarr = plt.subplots(rows,3,figsize=(12,rows*3))
        
            for i in range(rows):
                i_tracer = ipage*rows + i
                print('plotting: no.',i_tracer+1,self.tracername[i_tracer])
                
                # make variable names shorter for simplicity
                tracername = self.tracername[i_tracer]
                
                # assume v/v, convert to ppbv
                # might need modification in the future
                # TMS EDIT - PyGChem/iris save to NetCDF alters aixs order
                # Update from (time, longitude, latitude, model_level_number)
                # to: (time,lev,lat,lon)
#                data1 = self.data1[i_tracer][lev,:,:]#*1e9
#                data2 = self.data2[i_tracer][lev,:,:]#*1e9
                data1 = self.data1[i_tracer][:,:,lev].T#*1e9
                data2 = self.data2[i_tracer][:,:,lev].T#*1e9
#                unit self.data2_tracer_units
                unit = self.data2_tracer_units[tracername]
#                unit='ppbv'
                
                if plot_change:
                    # plot the change with respect to the initial condition,
                    # instead to the original field
                    # Update axis order 
#                    data0 = self.data0[i_tracer][lev,:,:]*1e9
                    data0 = self.data0[i_tracer][:,:,lev].T#*1e9
                    data1 -= data0
                    data2 -= data0
                    cmap=plt.cm.RdBu_r
                    
                else:
                    cmap=ga.WhGrYlRd
                
                if np.max(np.abs(data1)) < 1e-1 :
                    if unit=='ppbv':
                        data1 *= 1e3
                        data2 *= 1e3
                        unit='pptv'             
                elif np.max(np.abs(data1)) > 1e3 :
                    if unit=='ppbv':
                        data1 /= 1e3
                        data2 /= 1e3
                        unit='ppmv'
                                  
                # use the same scale for data1 and data2
                # use the scale of data2 by default.
                if switch_scale:
                    range_data = np.max(np.abs(data1))
                else:
                    range_data = np.max(np.abs(data2))
                # calculate the frac difference between two data sets
                data_ratio = (data1-data2)/data2
                range_ratio=np.max(np.abs(data_ratio))
                
                if plot_change:
                    vmin = -range_data
                    vmax = range_data
                    title= tracername+' change: '
                else:
                    vmin = 0
                    vmax = range_data
                    title= tracername+': '

                # overwrite much of above
                data1_min = np.min(np.abs(data1))
                data1_max = np.max(np.abs(data1))
                data2_min = np.min(np.abs(data2))
                data2_max = np.max(np.abs(data2))
                data_min =  np.min([data1_min,data2_min])
                data_max =  np.max([data1_max,data2_max]) 

                # always use max of both datasets instead
                vmax = data_max

                # use 0 for min unless dataset range is small 
                if (data_max-data_min)/data_max < 0.2:
                    vmin = data_min

                ga.tvmap(data1,axis=axarr[i,0],vmin=vmin,vmax=vmax,unit=unit,
                         cmap=cmap,title=title+self.model1,ticks = False)
                ga.tvmap(data2,axis=axarr[i,1],vmin=vmin,vmax=vmax,unit=unit,
                         cmap=cmap,title=title+self.model2,ticks = False)
                if full_range:
                    vmax=range_ratio
                    vmin=-range_ratio
                else:
                    vmax=2.0
                    vmin=-2.0
                ga.tvmap(data_ratio,
                         axis=axarr[i,2],
                         unit='unitless',
                         title=title+'('+self.model1+' - '+self.model2+') / '+self.model2,
                         ticks = False,
                         cmap=plt.cm.RdBu_r,
                         vmax=vmax,
                         vmin=vmin)
                
                if i_tracer+1 == N : 
                    i_hide = rows-1
                    while(i_hide > i):
                        [a.axis('off') for a in axarr[i_hide, :]]
                        i_hide -= 1
                    break # not using the full page
                
            fig.suptitle(self.longname+': '+tag,fontsize=15)

            print('saving one pdf page')
            pdf.savefig(fig)  # saves the figure into a pdf page
            plt.close() # close the current figure, prepare for the next page
            
        pdf.close() # close the pdf after saving all figures
        print(pdfname,' finished')

    def plot_zonal(self,mean=False,ilon=0,levs=None,
                   plot_change=False,switch_scale=False,
                   pdfname='default_zonalplot.pdf',tag='',ylog=False,
                   rows=3):
        
        '''
        Compare the zonal profiles of self.data1 and self.data2. Loop over all
        tracers in self.tracerlist. Plot on one multi-page pdf file. 
        
        Parameters
        ----------
        mean: logical, optional
            If set to True, then plot the zonal mean.
            Otherwise plot one cross-section specified by ilon
            
        ilon: integer, optional
            The longitude index of the cross-section. 
            Will have not effect mean is True.
            
        pdfname: string, optional but recommended
            The name the output pdf file
            
        tag: string, optional but recommended
            Will be shown as part of the title. For example,'zonal mean'
            
        rows: integer, optional
            How many rows on a page. Although the page size will be 
            adjusted accordingly, 3 rows look better.
            
        plot_change: logical, optional
            If set to True, then plot the change with respect to the initial
            condition (i.e. self.data0), rather than the original field.
            In this case, self.getdata0 must be executed before.
        
        switch_scale: logical, optional
            By default, the scale of data2 is used for both data1 and data2.
            This is because model2 is often regarded as the more correct one
            (the one for reference), so if model1 goes crazy, the color scale 
            will still make sense.
            
            If set to True, then use the scale of data1 instead. This is often
            combined with plot_change=True to more clearly show the regridding
            error.

        ylog: logical, optional
            By default, the y-axis is linear. Passing ylog=True enables
            log10 scale.
    
        Important Returns
        ----------
            A pdf file containing the plots of all tracers' zonal profile.
        
        '''
                
        N = len(self.tracername) # number of tracers
        
        Npages = (N-1)//rows+1 # have many pdf pages needed
        
        print('create ',pdfname)
        pdf=PdfPages(self.outputdir+pdfname) # create the pdf to save figures
        
        for ipage in range(Npages):
            
            fig, axarr = plt.subplots(rows,3,figsize=(12,rows*3))
        
            for i in range(rows):
                i_tracer = ipage*rows + i
                print('plotting: no.',i_tracer+1,self.tracername[i_tracer])
                
                # make variable names shorter for simplicity.
                # '*1.0' is needed, otherwise data1_3D amd self.data1 will 
                # share the same physical address, and will affect next plots.
                tracername = self.tracername[i_tracer]
                data1_3D = self.data1[i_tracer][:]*1.0
                data2_3D = self.data2[i_tracer][:]*1.0
                                     
                if plot_change:
                    # plot the change with respect to the initial condition,
                    # instead to the original field
                    data0_3D = self.data0[i_tracer][:]*1.0
                    data1_3D -= data0_3D
                    data2_3D -= data0_3D
                    cmap=plt.cm.RdBu_r
                else:
                    cmap=ga.WhGrYlRd
                
                # Pressure levels for GC levels 72:1, from GC wiki:
                # GEOS-Chem_vertical_grids#Vertical grids for GEOS-5, 
                # GEOS-FP, MERRA.2, and MERRA-2 
                # NOTE: eventually move this elsewhere! (ewl)
                levs72_hPa_rev = np.array([   
                        0.015,    0.026,   0.040,   0.057,   0.078,    
                        0.105,    0.140,   0.185,   0.245,   0.322, 
                        0.420,    0.546,   0.706,   0.907,   1.160, 
                        1.476,    1.868,   2.353,   2.948,   3.677, 
                        4.562,    5.632,   6.918,   8.456,  10.285, 
                        12.460,   15.050,  18.124,  21.761,  26.049, 
                        31.089,   36.993,  43.90,   52.016,  61.496, 
                        72.558,   85.439, 100.514, 118.250, 139.115, 
                        163.661,  192.587, 226.745, 267.087, 313.966, 
                        358.038,  396.112, 434.212, 472.335, 510.475, 
                        548.628,  586.793, 624.967, 663.146, 694.969, 
                        720.429,  745.890, 771.354, 796.822, 819.743, 
                        837.570,  852.852, 868.135, 883.418, 898.701, 
                        913.984,  929.268, 944.553, 959.837, 975.122, 
                        990.408, 1005.650 ])
                levs72_hPa = levs72_hPa_rev[::-1]

                # get limited levels if requested
                if levs is None:
                    lev = self.lev
                    press = levs72_hPa
                else:
                    lev = self.lev[levs[0]:levs[1]]
                    # TMS EDIT - PyGChem/iris save to NetCDF alters aixs order
                    # Update from (time, longitude, latitude, model_level_number)
                    # to: (time,lev,lat,lon)
#                    data1_3D=data1_3D[levs[0]:levs[1],:,:]
#                    data2_3D=data2_3D[levs[0]:levs[1],:,:]
                    data1_3D=data1_3D[:,:,levs[0]:levs[1]].T
                    data2_3D=data2_3D[:,:,levs[0]:levs[1]].T
                    press = levs72_hPa[levs[0]:levs[1]]

                # Reverse the y-axis in the plot so that P is decreasing up
                yrev = True
 
                if mean:
                    # get zonal mean
                    data1 = np.mean(data1_3D,axis=2)
                    data2 = np.mean(data2_3D,axis=2)
                else:
                    # get cross-section
                    data1 = data1_3D[:,:,ilon]
                    data2 = data2_3D[:,:,ilon]
                    
                # assume v/v, convert to ppbv
                # might need modification in the future
#                data1 *= 1e9
#                data2 *= 1e9
#                unit='ppbv'
                unit = self.data2_tracer_units[tracername]

                if np.max(data1) < 1e-1 :
                    if unit=='ppbv':
                        data1 *= 1e3
                        data2 *= 1e3
                        unit='pptv'                    
                elif np.max(data1) > 1e3 :
                    if unit=='ppbv':
                        data1 /= 1e3
                        data2 /= 1e3
                        unit='ppmv'
                    
                # use the same scale for data1 and data2
                # use the scale of data2 by default.
                if switch_scale:
                    range_data = np.max(np.abs(data1))
                else:
                    range_data = np.max(np.abs(data2))
                # calculate the difference between two data sets
                data_diff = data1-data2
                range_diff=np.max(np.abs(data_diff))
                
                if plot_change:
                    vmin = -range_data
                    vmax = range_data
                    title= tracername+' change: '
                else:
                    vmin = 0
                    vmax = range_data
                    title= tracername+': '
                         
                # overwrite much of above
                data1_min = np.min(np.abs(data1))
                data1_max = np.max(np.abs(data1))
                data2_min = np.min(np.abs(data2))
                data2_max = np.max(np.abs(data2))
                data_min =  np.min([data1_min,data2_min])
                data_max =  np.max([data1_max,data2_max]) 

                # always use max of both datasets instead
                vmax = data_max

                # use 0 for min unless dataset range is small 
                if (data_max-data_min)/data_max < 0.2:
                    vmin = data_min

                xlabel='lat'
                ylabel='hPa'
                          
                ga.tvplot(data1, axis=axarr[i,0], vmin=vmin, vmax=vmax,
                          unit=unit, cmap=cmap, x=self.lat, y=press,
                          yrev=yrev, xlabel=xlabel, ylabel=ylabel, ylog=ylog,
                          title=title+self.model1)
                ga.tvplot(data2, axis=axarr[i,1], vmin=vmin, vmax=vmax,
                          unit=unit, cmap=cmap, x=self.lat, y=press,
                          yrev=yrev, xlabel=xlabel, ylabel=ylabel, ylog=ylog,
                          title=title+self.model2)
                ga.tvplot(data_diff, axis=axarr[i,2], unit=unit,
                          x=self.lat, y=press, yrev=yrev,
                          xlabel=xlabel, ylabel=ylabel, ylog=ylog,
                          title=title+self.model1+' — '+self.model2,
                          cmap=plt.cm.RdBu_r,vmax=range_diff,vmin=-range_diff)
                
                # hide x ticks except the bottom plots
                if (i != rows-1 ) and ( i_tracer+1 != N) :
                    plt.setp([a.get_xticklabels() for a in axarr[i, :]], 
                             visible=False)
                    plt.setp([a.set_xlabel('') for a in axarr[i, :]], 
                             visible=False)
                
                if i_tracer+1 == N : 
                    if i != rows-1 :
                        [a.axis('off') for a in axarr[i+1, :]]
                    break # not using the full page
                
            #  hide y ticks for right plots
            plt.setp([a.get_yticklabels() for a in list(chain.from_iterable(axarr[:, 1:3]))], 
                     visible=False)
            plt.setp([a.set_ylabel('') for a in list(chain.from_iterable(axarr[:, 1:3]))],
                     visible=False)
                
            fig.suptitle(self.longname+': '+tag,fontsize=15)

            print('saving one pdf page')
            pdf.savefig(fig)  # saves the figure into a pdf page
            plt.close() # close the current figure, prepare for the next page
            
        pdf.close() # close the pdf after saving all figures
        print(pdfname,' finished')

    def plot_zonal_fracDiff(self,mean=False,ilon=0,levs=None,
                   plot_change=False,switch_scale=False,
                   pdfname='default_zonalplot.pdf',tag='',ylog=False,
                   full_range=False,rows=3):
        
        '''
        Compare the zonal profiles of self.data1 and self.data2. Loop over all
        tracers in self.tracerlist. Plot on one multi-page pdf file. 
        
        Parameters
        ----------
        mean: logical, optional
            If set to True, then plot the zonal mean.
            Otherwise plot one cross-section specified by ilon
            
        ilon: integer, optional
            The longitude index of the cross-section. 
            Will have not effect mean is True.
            
        pdfname: string, optional but recommended
            The name the output pdf file
            
        tag: string, optional but recommended
            Will be shown as part of the title. For example,'zonal mean'
            
        rows: integer, optional
            How many rows on a page. Although the page size will be 
            adjusted accordingly, 3 rows look better.
            
        plot_change: logical, optional
            If set to True, then plot the change with respect to the initial
            condition (i.e. self.data0), rather than the original field.
            In this case, self.getdata0 must be executed before.
        
        switch_scale: logical, optional
            By default, the scale of data2 is used for both data1 and data2.
            This is because model2 is often regarded as the more correct one
            (the one for reference), so if model1 goes crazy, the color scale 
            will still make sense.
            
            If set to True, then use the scale of data1 instead. This is often
            combined with plot_change=True to more clearly show the regridding
            error.

        ylog: logical, optional
            By default, the y-axis is linear. Passing ylog=True enables
            log10 scale.
    
        Important Returns
        ----------
            A pdf file containing the plots of all tracers' zonal profile.
        
        '''
                
        N = len(self.tracername) # number of tracers
        
        Npages = (N-1)//rows+1 # have many pdf pages needed
        
        print('create ',pdfname)
        pdf=PdfPages(self.outputdir+pdfname) # create the pdf to save figures
        
        for ipage in range(Npages):
            
            fig, axarr = plt.subplots(rows,3,figsize=(12,rows*3))
        
            for i in range(rows):
                i_tracer = ipage*rows + i
                print('plotting: no.',i_tracer+1,self.tracername[i_tracer])
                
                # make variable names shorter for simplicity.
                # '*1.0' is needed, otherwise data1_3D amd self.data1 will 
                # share the same physical address, and will affect next plots.
                tracername = self.tracername[i_tracer]
                data1_3D = self.data1[i_tracer][:]*1.0
                data2_3D = self.data2[i_tracer][:]*1.0
                                     
                if plot_change:
                    # plot the change with respect to the initial condition,
                    # instead to the original field
                    data0_3D = self.data0[i_tracer][:]*1.0
                    data1_3D -= data0_3D
                    data2_3D -= data0_3D
                    cmap=plt.cm.RdBu_r
                else:
                    cmap=ga.WhGrYlRd
                
                # Pressure levels for GC levels 72:1, from GC wiki:
                # GEOS-Chem_vertical_grids#Vertical grids for GEOS-5, 
                # GEOS-FP, MERRA.2, and MERRA-2 
                # NOTE: eventually move this elsewhere! (ewl)
                levs72_hPa_rev = np.array([   
                        0.015,    0.026,   0.040,   0.057,   0.078,    
                        0.105,    0.140,   0.185,   0.245,   0.322, 
                        0.420,    0.546,   0.706,   0.907,   1.160, 
                        1.476,    1.868,   2.353,   2.948,   3.677, 
                        4.562,    5.632,   6.918,   8.456,  10.285, 
                        12.460,   15.050,  18.124,  21.761,  26.049, 
                        31.089,   36.993,  43.90,   52.016,  61.496, 
                        72.558,   85.439, 100.514, 118.250, 139.115, 
                        163.661,  192.587, 226.745, 267.087, 313.966, 
                        358.038,  396.112, 434.212, 472.335, 510.475, 
                        548.628,  586.793, 624.967, 663.146, 694.969, 
                        720.429,  745.890, 771.354, 796.822, 819.743, 
                        837.570,  852.852, 868.135, 883.418, 898.701, 
                        913.984,  929.268, 944.553, 959.837, 975.122, 
                        990.408, 1005.650 ])
                levs72_hPa = levs72_hPa_rev[::-1]

                # get limited levels if requested
                if levs is None:
                    lev = self.lev
                    press = levs72_hPa
                else:
                    lev = self.lev[levs[0]:levs[1]]
                    data1_3D=data1_3D[levs[0]:levs[1],:,:]
                    data2_3D=data2_3D[levs[0]:levs[1],:,:]
                    press = levs72_hPa[levs[0]:levs[1]]

                # Reverse the y-axis in the plot so that P is decreasing up
                yrev = True
 
                if mean:
                    # get zonal mean
                    data1 = np.mean(data1_3D,axis=2)
                    data2 = np.mean(data2_3D,axis=2)
                else:
                    # get cross-section
                    data1 = data1_3D[:,:,ilon]
                    data2 = data2_3D[:,:,ilon]
                    
                # assume v/v, convert to ppbv
                # might need modification in the future
                data1 *= 1e9
                data2 *= 1e9
                unit='ppbv'

                if np.max(data1) < 1e-1 :
                    data1 *= 1e3
                    data2 *= 1e3
                    unit='pptv'                    
                elif np.max(data1) > 1e3 :
                    data1 /= 1e3
                    data2 /= 1e3
                    unit='ppmv'
                    
                # use the same scale for data1 and data2
                # use the scale of data2 by default.
                if switch_scale:
                    range_data = np.max(np.abs(data1))
                else:
                    range_data = np.max(np.abs(data2))
                
                # calculate the frac difference between two data sets
                data_ratio = (data1-data2)/data2
                range_ratio=np.max(np.abs(data_ratio))
                ## calculate the difference between two data sets
                #data_diff = data1-data2
                #range_diff=np.max(np.abs(data_diff))
                
                if plot_change:
                    vmin = -range_data
                    vmax = range_data
                    title= tracername+' change: '
                else:
                    vmin = 0
                    vmax = range_data
                    title= tracername+': '
                         
                # overwrite much of above
                data1_min = np.min(np.abs(data1))
                data1_max = np.max(np.abs(data1))
                data2_min = np.min(np.abs(data2))
                data2_max = np.max(np.abs(data2))
                data_min =  np.min([data1_min,data2_min])
                data_max =  np.max([data1_max,data2_max]) 

                # always use max of both datasets instead
                vmax = data_max

                # use 0 for min unless dataset range is small 
                if (data_max-data_min)/data_max < 0.2:
                    vmin = data_min

                xlabel='lat'
                ylabel='hPa'
                          
                ga.tvplot(data1, axis=axarr[i,0], vmin=vmin, vmax=vmax,
                          unit=unit, cmap=cmap, x=self.lat, y=press,
                          yrev=yrev, xlabel=xlabel, ylabel=ylabel, ylog=ylog,
                          title=title+self.model1)
                ga.tvplot(data2, axis=axarr[i,1], vmin=vmin, vmax=vmax,
                          unit=unit, cmap=cmap, x=self.lat, y=press,
                          yrev=yrev, xlabel=xlabel, ylabel=ylabel, ylog=ylog,
                          title=title+self.model2)
                if full_range:
                    vmax=range_ratio
                    vmin=-range_ratio
                else:
                    vmax=2.0
                    vmin=-2.0
                ga.tvplot(data_ratio, axis=axarr[i,2], unit=unit,
                          x=self.lat, y=press, yrev=yrev,
                          xlabel=xlabel, ylabel=ylabel, ylog=ylog,
                          title=title+self.model1+' — '+self.model2,
                          cmap=plt.cm.RdBu_r,vmax=vmax,vmin=vmin)
                
                # hide x ticks except the bottom plots
                if (i != rows-1 ) and ( i_tracer+1 != N) :
                    plt.setp([a.get_xticklabels() for a in axarr[i, :]], 
                             visible=False)
                    plt.setp([a.set_xlabel('') for a in axarr[i, :]], 
                             visible=False)
                
                if i_tracer+1 == N : 
                    if i != rows-1 :
                        [a.axis('off') for a in axarr[i+1, :]]
                    break # not using the full page
                
            #  hide y ticks for right plots
            plt.setp([a.get_yticklabels() for a in list(chain.from_iterable(axarr[:, 1:3]))], 
                     visible=False)
            plt.setp([a.set_ylabel('') for a in list(chain.from_iterable(axarr[:, 1:3]))],
                     visible=False)
                
            fig.suptitle(self.longname+': '+tag,fontsize=15)

            print('saving one pdf page')
            pdf.savefig(fig)  # saves the figure into a pdf page
            plt.close() # close the current figure, prepare for the next page
            
        pdf.close() # close the pdf after saving all figures
        print(pdfname,' finished')
        
        
    def plot_all(self,
                 plot_surf=True,
                 plot_surf_fracdiff=True,
                 plot_500hpa=True,
                 plot_500hpa_fracdiff=True,
                 plot_zonal_lowertrop_180lon_diff=True,
                 plot_zonal_lowertrop_180lon_fracdiff=True,
                 plot_zonal_lowertrop_mean_diff=True,
                 plot_zonal_lowertrop_mean_fracdiff=True,
                 plot_zonal_trop_180lon_diff=True,
                 plot_zonal_trop_180lon_fracdiff=True,
                 plot_zonal_trop_mean_diff=True,
                 plot_zonal_trop_mean_fracdiff=True,
                 plot_zonal_strat_180lon_diff=True,
                 plot_zonal_strat_180lon_fracdiff=True,
                 plot_zonal_strat_mean_diff=True,
                 plot_zonal_strat_mean_fracdiff=True,
                 plot_change=False):
        '''
        Just to wrap several plot_layer and plot_zonal calls for convenience.
        
        Parameters
        ----------
        plot_change: logical, optional
            If set to True, then plot the change with respect to the initial
            condition (i.e. self.data0), rather than the original field.
            In this case, self.getdata0 must be executed before.
        
        plot_surf,plot_500hpa,plot_zonal: logical, optional
            plot that feature or not. Default is to plot everything.
        
        Important Returns
        ----------
            Several pdf files.
        '''

        if plot_change:
            switch_scale=True # use data1's scale when plotting change
            tag='_change' # the file name should be slightly different
        else:
            switch_scale=False
            tag=''
        
        if plot_surf:
            self.plot_layer(pdfname=self.shortname+tag
                            +'_surf_diff.pdf',
                            lev=0,
                            tag='surface',
                            switch_scale=switch_scale,
                            plot_change=plot_change)
        if plot_surf_fracdiff:
            self.plot_fracDiff(pdfname=self.shortname+tag
                               +'_surf_fracDiff.pdf',
                            lev=0,
                            tag='surface',
                            switch_scale=switch_scale,
                            full_range=False,
                            plot_change=plot_change)
        if plot_500hpa:
            self.plot_layer(pdfname=self.shortname+tag
                            +'_500hpa_diff.pdf',
                            lev=22,
                            tag='500 hpa',
                            switch_scale=switch_scale,
                            plot_change=plot_change)
        if plot_500hpa_fracdiff:
            self.plot_fracDiff(pdfname=self.shortname+tag
                               +'_500hPa_fracDiff.pdf',
                            lev=22,
                            tag='500 hpa',
                            switch_scale=switch_scale,
                            full_range=False,
                            plot_change=plot_change)

        if plot_zonal_lowertrop_180lon_diff:
            self.plot_zonal(pdfname=self.shortname+tag
                            +'_lowertrop_180lon.pdf',
                            ilon=0,
                            levs=[0,21],
                            tag='180 lon lower trop',
                            switch_scale=switch_scale,
                            plot_change=plot_change)
        if plot_zonal_lowertrop_mean_diff:
            self.plot_zonal(pdfname=self.shortname+tag
                            +'_lowertrop_zonalmean.pdf',
                            mean=True,
                            levs=[0,21],
                            tag='zonal mean lower trop',
                            switch_scale=switch_scale,
                            plot_change=plot_change)
        if plot_zonal_trop_180lon_diff:
            self.plot_zonal(pdfname=self.shortname+tag
                            +'_trop_180lon.pdf',
                            ilon=0,
                            levs=[0,39],
                            tag='180 lon trop',
                            switch_scale=switch_scale,
                            plot_change=plot_change)
        if plot_zonal_trop_mean_diff:
            self.plot_zonal(pdfname=self.shortname+tag
                            +'_trop_zonalmean.pdf',
                            mean=True,
                            levs=[0,39],
                            tag='zonal mean trop',
                            switch_scale=switch_scale,
                            plot_change=plot_change)

        if plot_zonal_strat_180lon_diff:
            self.plot_zonal(pdfname=self.shortname+tag
                            +'_strat_180lon.pdf',
                            ilon=0,
                            levs=[34,71],
                            tag='180 lon strat',
                            switch_scale=switch_scale,
                            plot_change=plot_change,
                            ylog=True)
        if plot_zonal_strat_mean_diff:
            self.plot_zonal(pdfname=self.shortname+tag
                            +'_strat_zonalmean.pdf',
                            mean=True,
                            levs=[34,71],
                            tag='zonal mean strat',
                            switch_scale=switch_scale,
                            plot_change=plot_change,
                            ylog=True)
        if plot_zonal_trop_180lon_fracdiff:
            self.plot_zonal_fracDiff(pdfname=self.shortname+tag
                                     +'_trop_180lon_fracDiff.pdf',
                            ilon=0,
                            levs=[0,39],
                            tag='180 lon trop',
                            switch_scale=switch_scale,
                            full_range=False,
                            plot_change=plot_change)
        if plot_zonal_trop_mean_fracdiff:
            self.plot_zonal_fracDiff(pdfname=self.shortname+tag
                                     +'_trop_zonalmean_fracDiff.pdf',
                            mean=True,
                            levs=[0,39],
                            tag='zonal mean trop',
                            switch_scale=switch_scale,
                            full_range=False,
                            plot_change=plot_change)
        if plot_zonal_strat_180lon_fracdiff:
            self.plot_zonal_fracDiff(pdfname=self.shortname+tag
                                     +'_strat_180lon_fracDiff.pdf',
                            ilon=0,
                            levs=[34,71],
                            tag='180 lon strat',
                            switch_scale=switch_scale,
                            plot_change=plot_change,
                            full_range=False,
                                     ylog=True)
        if plot_zonal_strat_mean_fracdiff:
            self.plot_zonal_fracDiff(pdfname=self.shortname+tag
                                     +'_strat_zonalmean_fracDiff.pdf',
                            mean=True,
                            levs=[34,71],
                            tag='zonal mean strat',
                            switch_scale=switch_scale,
                            plot_change=plot_change,
                            full_range=False,
                            ylog=True)



            

                
