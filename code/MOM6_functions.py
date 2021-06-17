'''
    Functions for use with global MOM6 output and regional Indian Ocean output.
    
    Created by Jenna Pearson, Last updated 06/2021
'''

# import parameters used

from pars import *
#===============================================================================================================

def shiftlon_glob_MOM6_0_360(lat,lon,invar, type = 'tracer',plot = False, grid = True):  
    '''
        Shifts global MOM6 grid and a given variable from -300 to 60 --> 0 to 360 
        so that the Indian Ocean is not cut in half.
            type = 'tracer -> dims (yh,xh)   [default]
            type = 'u' --> dims (yh,xq)
            type = 'v' --> dims (yq,xh)
        
        Returns numpy array of shifted variable, and lat and lon if grid == True.
    '''
    # packages
    import numpy as np
    import matplotlib.pyplot as plt

    # make sure all xarray datasets are converted to numpy arrays
    lon = np.array(lon)
    lat = np.array(lat)
    invar = np.array(invar)
    
    if plot:
        cmin = np.nanmin(invar)
        cmax = np.nanmax(invar)
        
        fig = plt.figure(figsize=(16, 4), dpi = 200)
        
        ax = fig.add_subplot(121)
        ax.pcolor(lon,lat,invar[0,:,:],vmin = cmin, vmax = cmax)
        ax.set_title('Original Grid at First Time-step')
        ax.set_xlabel('Longitude (Degrees)')
        ax.set_ylabel('Latitude (Degrees)')

    if type == 'u': # remove last piece because end points are periodic/the same for u only
        lon = lon[1:,:]
        lat = lat[1:,:]
        invar = invar[:,1:,:]
        
    # set -300 to 0 to be positive 60-360
    lon[lon<0] = lon[lon<0]+360
    
    # sort data
    sortind = np.argsort(lon,axis = 1)
    lon = np.take_along_axis(lon, sortind, axis=1)
    lat = np.take_along_axis(lat,sortind, axis=1)

    for tt in range(invar.shape[0]):
        invar[tt,:,:] = np.take_along_axis(invar[tt,:,:],sortind,axis=1)
        
    if type == 'u':
        # add on another column to preserve dimensions
        lon = np.c_[lon,lon[:,0]+360]
        lat = np.c_[lat,lat[:,0]]
        invar = np.dstack((invar,invar[:,:,0]))
        
    if plot:
        ax = fig.add_subplot(122)
        ax.pcolor(lon,lat,invar[0,:,:],vmin = cmin, vmax = cmax)
        ax.set_title('Shifted Grid at First Time-step')    
        ax.set_xlabel('Longitude (Degrees)')
    
    if grid == True:
        return lat,lon,invar
    else:
        return invar
    
#===============================================================================================================

def shiftlon_cobalt_ds(ds_in, plot = False):  
    '''
        Shifts global MOM6 grid and cobalt tracers from -300 to 60 --> 0 to 360 
        so that the Indian Ocean is not cut in half.dims (yh,xh)
        
        Input is an xarray dataset containing variables to be shifted as well as 
        Returns xarray dataset of shifted variables and plots 2d grid if plot == True.
    '''
    # packages
    import numpy as np
    import matplotlib.pyplot as plt
    import xarray as xr
    
    # intialize output dataset
    
    ds_out = xr.Dataset(coords={})

    for var,vv in enumerate(list(ds_in.data_vars)):
        # make sure all xarray datasets are converted to numpy arrays
        lon = np.array(ds_in.geolon)
        lat = np.array(ds_in.geolat)
        invar = np.array(var)

        if (plot == True) & (vv == 0):
            cmin = np.nanmin(invar)
            cmax = np.nanmax(invar)

            fig = plt.figure(figsize=(16, 4), dpi = 200)

            ax = fig.add_subplot(121)
            ax.pcolor(lon,lat,invar[0,:,:],vmin = cmin, vmax = cmax)
            ax.set_title('Original Grid at First Time-step')
            ax.set_xlabel('Longitude (Degrees)')
            ax.set_ylabel('Latitude (Degrees)')

        # set -300 to 0 to be positive 60-360
        lon[lon<0] = lon[lon<0]+360

        # sort data
        sortind = np.argsort(lon,axis = 1)
        lon = np.take_along_axis(lon, sortind, axis=1)
        lat = np.take_along_axis(lat,sortind, axis=1)

        if invar.ndim == 3:
            for tt in range(invar.shape[0]):
                invar[tt,:,:] = np.take_along_axis(invar[tt,:,:],sortind,axis=1)
        elif invar.ndim == 4:
            for tt in range(invar.shape[0]):
                for dd in range(invar.shape[1]):
                    invar[tt,dd,:,:] = np.take_along_axis(invar[tt,dd,:,:],sortind,axis=1)
        else: 
            print('Dimensions must be 3 or 4.')
  
        if (plot == True) & (vv == 0):
            ax = fig.add_subplot(122)
            ax.pcolor(lon,lat,invar[0,:,:],vmin = cmin, vmax = cmax)
            ax.set_title('Shifted Grid at First Time-step')    
            ax.set_xlabel('Longitude (Degrees)')

    # add to dataset
    return ds_out
#===============================================================================================================

def subset_global_MOM6_Indian_Ocean(lat,lon,invar, type = 'tracer',plot = False,grid = True):
    '''
        Subset global MOM6 grid for the Indian Ocean. 
            type = 'tracer -> dims (yh,xh)   [default]
            type = 'u' --> dims (yh,xq)
            type = 'v' --> dims (yq,xh)
        
        Returns numpy array of subsetted variable, and lat and lon if grid == True.
        
        Optional boolean plot variable to show domain before and after subset.
        
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    
    invar = np.array(invar)
    lat = np.array(lat)
    lon = np.array(lon)
    
    if plot:
        cmin = np.nanmin(invar)
        cmax = np.nanmax(invar)
        
        fig = plt.figure(figsize=(16, 4), dpi = 200)
        
        ax = fig.add_subplot(121)
        ax.pcolor(lon,lat,invar[0,:,:],vmin = cmin, vmax = cmax)
        ax.set_title('Original Grid')
        ax.set_xlabel('Longitude (Degrees)')
        ax.set_ylabel('Latitude (Degrees)')
        

    if type == 'tracer':
        lat1 = 375
        lon1 = 117
        lat = lat[lat1:lat1+256,lon1:lon1+428]
        lon = lon[lat1:lat1+256,lon1:lon1+428]
        
        #subset based on number of dimensions
        if invar.ndim == 4: # time, depth, lat, lon
            invar = invar[:,:,lat1:lat1+256,lon1:lon1+428]
        elif invar.ndim == 3: # time, lat, lon
            invar = invar[:,lat1:lat1+256,lon1:lon1+428]
        
    elif type == 'u':
        lat1 = 375
        lon1 = 117
        lat = lat[lat1:lat1+256,lon1:lon1+429]
        lon = lon[lat1:lat1+256,lon1:lon1+429]
        
        #subset based on number of dimensions
        if invar.ndim == 4: # time, depth, lat, lon
            invar = invar[:,:,lat1:lat1+256,lon1:lon1+429]
        elif invar.ndim == 3: # time, lat, lon
            invar = invar[:,lat1:lat1+256,lon1:lon1+429]
            
    elif type == 'v':
        lat1 = 375
        lon1 = 117
        lat = lat[lat1:lat1+257,lon1:lon1+428]
        lon = lon[lat1:lat1+257,lon1:lon1+428]
        
        #subset based on number of dimensions
        if invar.ndim == 4: # time, depth, lat, lon
            invar = invar[:,:,lat1:lat1+257,lon1:lon1+428]
        elif invar.ndim == 3: # time, lat, lon
            invar = invar[:,lat1:lat1+257,lon1:lon1+428]
    
    if plot:
        ax = fig.add_subplot(122)
        ax.pcolor(lon,lat,invar[0,:,:],vmin = cmin, vmax = cmax)
        ax.set_title('Subsetted Grid')
        ax.set_xlabel('Longitude (Degrees)')
        ax.set_ylabel('Latitude (Degrees)')
    if grid == True:
        return lat, lon, invar
    else: 
        return invar
   
#===============================================================================================================
    
# def surface_vorticity(SSU,SSU,geolat_u,geolon_u,geolat_v,geolon_v, cont_check = False):
#     '''
#     Takes n by m grid of lat/ lon and determines centered differences in the interior and backwards and
#     forwards differences at the boundaries
    
#     Returns a matrix the same size as lat and lon.
    
#     Optional continuity check, should be close to zero, to make sure the derivatives work as they should. 
    
#     Make sure lat increases from top row to bottom row and lon increases from left column to right column so that the 
#     signs of dx and dy are correct - fix this later to add a check in function and adjust 
#     '''
    
#     from geopy.distance import geodesic
#     import numpy as np
#     import itertools

#     n,m = lat.shape
#     du_y
#     dv_x
#     dx
#     dy = np.full(lat.shape,np.nan),np.full(lat.shape,np.nan),np.full(lat.shape,np.nan),np.full(lat.shape,np.nan)

#     for ii,jj in itertools.product(range(n),range(m)):
#         if (ii == 0) & (jj == 0): # upper left corner
#             dx[ii,jj] = geodesic((lat[ii,jj],lon[ii,jj]),(lat[ii,jj+1],lon[ii,jj+1])).m
#             dy[ii,jj] = geodesic((lat[ii,jj],lon[ii,jj]),(lat[ii+1,jj],lon[ii+1,jj])).m
#             du_y[ii,jj] = u[ii+1,jj]-u[ii,jj]
#             du_x[ii,jj] = u[ii,jj+1]-u[ii,jj]
#             dv_x[ii,jj] = v[ii,jj+1]-v[ii,jj]
#             dv_y[ii,jj] = v[ii+1,jj]-v[ii,jj]
#         elif (ii == 0) & (jj == (m-1)): # upper right corner
#             dx[ii,jj] = geodesic((lat[ii,jj],lon[ii,jj]),(lat[ii,jj-1],lon[ii,jj-1])).m
#             dy[ii,jj] = geodesic((lat[ii,jj],lon[ii,jj]),(lat[ii+1,jj],lon[ii+1,jj])).m
#             du_y[ii,jj] = u[ii+1,jj]-u[ii,jj]
#             du_x[ii,jj] = u[ii,jj]-u[ii,jj-1]
#             dv_x[ii,jj] = v[ii,jj]-v[ii,jj-1]
#             dv_y[ii,jj] = v[ii+1,jj]-v[ii,jj]
#         elif (ii == (n-1)) & (jj == 0): # lower left corner
#             dx[ii,jj] = geodesic((lat[ii,jj],lon[ii,jj]),(lat[ii,jj+1],lon[ii,jj+1])).m
#             dy[ii,jj] = geodesic((lat[ii,jj],lon[ii,jj]),(lat[ii-1,jj],lon[ii-1,jj])).m
#             du_y[ii,jj] = u[ii,jj]-u[ii-1,jj]
#             du_x[ii,jj] = u[ii,jj+1]-u[ii,jj]
#             dv_x[ii,jj] = v[ii,jj+1]-v[ii,jj]
#             dv_y[ii,jj] = v[ii,jj]-v[ii-1,jj]
#         elif (ii == (n-1)) & (jj == (m-1)): # lower right corner
#             dx[ii,jj] = geodesic((lat[ii,jj],lon[ii,jj]),(lat[ii,jj-1],lon[ii,jj-1])).m
#             dy[ii,jj] = geodesic((lat[ii,jj],lon[ii,jj]),(lat[ii-1,jj],lon[ii-1,jj])).m
#             du_y[ii,jj] = u[ii,jj]-u[ii-1,jj]
#             du_x[ii,jj] = u[ii,jj]-u[ii,jj-1]
#             dv_x[ii,jj] = v[ii,jj]-v[ii,jj-1]
#             dv_y[ii,jj] = v[ii,jj]-v[ii-1,jj]
#         elif ii == 0: # upper row interior cols
#             dx[ii,jj] = geodesic((lat[ii,jj-1],lon[ii,jj-1]),(lat[ii,jj+1],lon[ii,jj+1])).m
#             dy[ii,jj] = geodesic((lat[ii,jj],lon[ii,jj]),(lat[ii+1,jj],lon[ii+1,jj])).m
#             du_y[ii,jj] = u[ii+1,jj]-u[ii,jj]
#             du_x[ii,jj] = u[ii,jj+1]-u[ii,jj-1]
#             dv_x[ii,jj] = v[ii,jj+1]-v[ii,jj-1]
#             dv_y[ii,jj] = v[ii+1,jj]-v[ii,jj]
#         elif ii == (n-1): # lower row interior col s
#             dx[ii,jj] = geodesic((lat[ii,jj-1],lon[ii,jj-1]),(lat[ii,jj+1],lon[ii,jj+1])).m
#             dy[ii,jj] = geodesic((lat[ii,jj],lon[ii,jj]),(lat[ii-1,jj],lon[ii-1,jj])).m
#             du_y[ii,jj] = u[ii,jj]-u[ii-1,jj]
#             du_x[ii,jj] = u[ii,jj+1]-u[ii,jj-1]
#             dv_x[ii,jj] = v[ii,jj+1]-v[ii,jj-1]
#             dv_y[ii,jj] = v[ii,jj]-v[ii-1,jj]
#         elif jj == 0: # left column interior rows
#             dx[ii,jj] = geodesic((lat[ii,jj],lon[ii,jj]),(lat[ii,jj+1],lon[ii,jj+1])).m
#             dy[ii,jj] = geodesic((lat[ii-1,jj],lon[ii-1,jj]),(lat[ii+1,jj],lon[ii+1,jj])).m
#             du_y[ii,jj] = u[ii+1,jj]-u[ii-1,jj]
#             du_x[ii,jj] = u[ii,jj+1]-u[ii,jj]
#             dv_x[ii,jj] = v[ii,jj+1]-v[ii,jj]
#             dv_y[ii,jj] = v[ii+1,jj]-v[ii-1,jj]
#         elif jj == (m-1): # right column interior rows
#             dx[ii,jj] = geodesic((lat[ii,jj],lon[ii,jj]),(lat[ii,jj-1],lon[ii,jj-1])).m
#             dy[ii,jj] = geodesic((lat[ii-1,jj],lon[ii-1,jj]),(lat[ii+1,jj],lon[ii+1,jj])).m
#             du_y[ii,jj] = u[ii+1,jj]-u[ii-1,jj]
#             du_x[ii,jj] = u[ii,jj]-u[ii,jj-1]
#             dv_x[ii,jj] = v[ii,jj]-v[ii,jj-1]
#             dv_y[ii,jj] = v[ii+1,jj]-v[ii-1,jj]
#         else: # completely interior
#             dx[ii,jj] = geodesic((lat[ii,jj-1],lon[ii,jj-1]),(lat[ii,jj+1],lon[ii,jj+1])).m 
#             dy[ii,jj] = geodesic((lat[ii-1,jj],lon[ii-1,jj]),(lat[ii+1,jj],lon[ii+1,jj])).m
#             du_y[ii,jj] = u[ii+1,jj]-u[ii-1,jj]
#             du_x[ii,jj] = u[ii,jj+1]-u[ii,jj-1]
#             dv_x[ii,jj] = v[ii,jj+1]-v[ii,jj-1]
#             dv_y[ii,jj] = v[ii+1,jj]-v[ii-1,jj]

#     du_dy = du_y/dy
#     du_dx = du_x/dx
#     dv_dx = dv_x/dx
#     dv_dy = dv_y/dy
#     vort = dv_dx-du_dy
    
#     # optional continuity check
#     du_dx + dv_dy
    
#     # plot
            
#     return np.array(vort),np.array(du_dy), np.array(dv_dx)