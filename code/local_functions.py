cbounds = [48.5,105,-1.5,33]

#===============================================================================================================

def as_si(x, ndp):
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))

#===============================================================================================================

def get_default_args(func):
    import inspect
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

#===============================================================================================================


def ylabel_map(ax,label,x = -0.15, y = 0.5, fontsize = 18, color = 'k'):
    ax.text(x, y, label, va='bottom', ha='center',color = color,
        rotation='vertical', rotation_mode='anchor',
        transform=ax.transAxes, fontsize = fontsize)
    
#===============================================================================================================


def o2sat(temp,psal):
    ''' 
        CALCULATE OXYGEN CONCENTRATION AT SATURATION f(T,S)
        
        https://www.mbari.org/products/research-software/matlab-scripts-oceanographic-calculations/
        and python code found here:
        https://github.com/kallisons/pO2_conversion/blob/master/pycode/function_pO2.ipynb
        
        Code is based on: 
        Garcia and Gordon (1992) oxygen solubility in seawater, better fitting equations. L&O 37: 1307-1312
        using the coefficients for umol/kg from the combined fit column of Table 1
      
        Input:   temp = temperature (degree C)
                 sal  = practical salinity (PSS-78)
                 
        Output:  Oxygen staturation at one atmosphere (umol/kg).
        
    '''
    import numpy as np
    
    a_0 =  5.80818;
    a_1 =  3.20684;
    a_2 =  4.11890;
    a_3 =  4.93845;
    a_4 =  1.01567;
    a_5 =  1.41575;
  
    b_0 = -7.01211e-03;
    b_1 = -7.25958e-03;
    b_2 = -7.93334e-03;
    b_3 = -5.54491e-03;
  
    c_0 = -1.32412e-07;
  
    ts = np.log((298.15 - temp) / (273.15 + temp))

    A = a_0 + a_1*ts + a_2*ts**2 + a_3*ts**3 + a_4*ts**4 + a_5*ts**5 
               
    B = psal*(b_0 + b_1*ts + b_2*ts**2 + b_3*ts**3)
               
    O2_sat = np.exp(A + B + c_0*psal**2)
    
    return O2_sat

#===============================================================================================================

def add_single_vert_cbar(fig,p,label, extend = 'neither', loc=[0.925, 0.125, 0.015, 0.75]):
    cbar_ax = fig.add_axes(loc)
    cbar = fig.colorbar(p,cax=cbar_ax, pad=0.04, extend = extend)
    cbar.set_label(label)
    return cbar
    
#===============================================================================================================

def add_text(ax, text, x = 0.01, y = .945, fontsize = 12, color = 'k', weight = 'normal', rotation = 0, style = 'normal'):
    ax.annotate(text, xy=(x,y), xycoords="axes fraction", fontsize = fontsize, color = color, style = style,
               weight=weight, rotation = rotation)
    return None

#===============================================================================================================

def add_letter(ax, letter, x = 0.01, y = .945, fontsize = 12, weight='bold', color = 'k'):
    ax.annotate(letter, xy=(x,y), xycoords="axes fraction", fontsize = fontsize, weight='bold', color = color)
    return None

#===============================================================================================================

def add_land(ax,bounds = cbounds, countries = False, rivers = False, lakes = False, facecolor = 'w',
             lcolor='dimgray',ccolor = '#878787',rcolor = 'cyan'):
#             lcolor = '#b5651d',ccolor = '#ca9852',rcolor = '#3944bc'):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    import cartopy.feature as cfeature
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                            edgecolor='face')
    ax.add_feature(land,color=lcolor,zorder = 1) # #b5651d
    ax.background_patch.set_facecolor(facecolor)
#     ax.coastlines(resolution='50m',zorder = 2, color = 'gray)
    if countries == True:
        countries_10m = cfeature.NaturalEarthFeature('cultural', 'admin_0_countries', '10m')
        ax.add_feature(countries_10m,facecolor='None', edgecolor=ccolor, linewidth=0.5) # #65350F
#         ax.add_feature(cfeature.BORDERS)
    if rivers == True:
        rivers_10m = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '10m')
        ax.add_feature(rivers_10m, facecolor='None', edgecolor=rcolor, linewidth=0.25) # '#404040'
#         ax.add_feature(cfeature.RIVERS)
    if lakes == True:
        ax.add_feature(cfeature.LAKES, alpha=0.5)
    
    g = ax.gridlines(draw_labels=True,alpha=0)
    g.xlabels_top = False
    g.ylabels_right = False
    g.xlabel_style = {'size': 15}
    g.ylabel_style = {'size': 15}
    g.xformatter = LONGITUDE_FORMATTER
    g.yformatter = LATITUDE_FORMATTER
    ax.axes.axis('tight')
    ax.set_extent(bounds, crs=ccrs.PlateCarree())
    ax.outline_patch.set_linewidth(0.5)
    return g

#===============================================================================================================

def add_bathy(ax,bounds = cbounds, zorder = 0):
    # datasets: https://www.naturalearthdata.com/downloads/10m-physical-vectors/
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    import matplotlib as mpl 
    
    cmap = plt.get_cmap('Blues')
    norm = mpl.colors.Normalize(0, 10000)

    for letter, level in [
                      ('L', 0),
                      ('K', 200),
                      ('J', 1000),
                      ('I', 2000),
                      ('H', 3000),
                      ('G', 4000),
                      ('F', 5000),
                      ('E', 6000),
                      ('D', 7000),
                      ('C', 8000),
                      ('B', 9000),
                      ('A', 10000)]:
        
        
        bathym = cfeature.NaturalEarthFeature(name='bathymetry_{}_{}'.format(letter, level),
                                     scale='10m', category='physical')
        ax.add_feature(bathym, facecolor=cmap(norm(level)), edgecolor='face',zorder = zorder)
    ax.axes.axis('tight')
    ax.set_extent(bounds, crs=ccrs.PlateCarree())
    return None

#===============================================================================================================

# still working on this
def add_bathy_clines(ax,bounds = cbounds, lmax = 10000, linewidth = 2):
    # datasets: https://www.naturalearthdata.com/downloads/10m-physical-vectors/
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    import matplotlib as mpl 
    from shapely.ops import cascaded_union
    
    cmap = plt.get_cmap('Blues')
    norm = mpl.colors.Normalize(0, 10000)

    name_list = [('L', 0),
            ('K', 200),
            ('J', 1000),
            ('I', 2000),
            ('H', 3000),
            ('G', 4000),
            ('F', 5000),
            ('E', 6000),
            ('D', 7000),
            ('C', 8000),
            ('B', 9000),
            ('A', 10000)]
    
    letter = [item for item in name_list if item[1] <= lmax]
    
    for letter,level in name_list:
        
        bathym = cfeature.NaturalEarthFeature(name='bathymetry_{}_{}'.format(letter, level), scale='10m', category='physical')
        bathym = cascaded_union(list(bathym.geometries()))
        ax.add_geometries(bathym, facecolor='none', edgecolor='black', linestyle='solid', linewidth=linewidth, crs=ccrs.PlateCarree())
    
    ax.axes.axis('tight')
    ax.set_extent(bounds, crs=ccrs.PlateCarree())
    return None

#===============================================================================================================

def add_bathy_cbar(fig,ax,label='m below sea level',pos = [0.13, .18, 0.23, 0.03],lmax=10000):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    
    cmap = plt.get_cmap('Blues')
    norm = mpl.colors.Normalize(0, 10000)
    
    levellist = np.array([0,200,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000])
    levellist = levellist[levellist<=lmax]
    levellist = list(levellist)
    
    colorlist = []
    bounds = []
    for level in levellist:
        colorlist.append(cmap(norm(level)))
        bounds.append(level)

    cmap = mpl.colors.ListedColormap(colorlist)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    bnd = np.array(bounds)
    cbar_ax = fig.add_axes(pos)
    cb2 = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap,
                                    norm=norm,
                                    boundaries=bounds,
                                    ticks=(bnd[1:]-bnd[:-1])/2 + bnd[:-1],
#                                     spacing='proportional',
                                    orientation='horizontal')
    cb2.set_label(label)
    cb2.set_ticklabels(bounds[1:])
    return cb2

#===============================================================================================================

def add_single_bathy_cline(ax, level,linewidth = 2, color = 'k'):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from shapely.ops import cascaded_union

    name_list = [('L', 0),
            ('K', 200),
            ('J', 1000),
            ('I', 2000),
            ('H', 3000),
            ('G', 4000),
            ('F', 5000),
            ('E', 6000),
            ('D', 7000),
            ('C', 8000),
            ('B', 9000),
            ('A', 10000)]
    
    name_list = [item for item in name_list if item[1] ==level]
    
    bathym = cfeature.NaturalEarthFeature(name='bathymetry_{}_{}'.format(name_list[0][0], name_list[0][1]), scale='10m', category='physical')
    bathym = cascaded_union(list(bathym.geometries()))
    ax.add_geometries(bathym, facecolor='none', edgecolor=color, linestyle='solid', linewidth=linewidth, crs=ccrs.PlateCarree())
    
#===============================================================================================================

def add_box(ax,box_bounds,clrs, fill = False, linewidth = 2):
    
    # lonmin,lonmax, latmin, latmax 
    from matplotlib.patches import Rectangle
    for ii, box in enumerate(box_bounds): 
        p = Rectangle(
        (box[0], box[2]), box[1]-box[0], box[3]-box[2],
        linewidth=linewidth,fill=False,color=clrs[ii], zorder = 3)

        ax.add_patch(p)
        rx, ry = p.get_xy()
        cx = rx + p.get_width()/2.0
        cy = ry + p.get_height()/2.0

#===============================================================================================================
        
def latlon_2D_vorticity(u,v,lat,lon):
    '''
    takes n by m grid of lat/ lon and determines centered differeneces in the interior and backwards and forwards differences 
    at the boundaries
    
    returns a matrix the same size as lat and lon
    make sure lat increases from top row to bottom row and lon increases from left column to right column so that the 
    signs of dx and dy are correct
    '''
    
    from geopy.distance import geodesic
    import numpy as np
    import itertools

    n,m = lat.shape
    du_y,dv_x,dx,dy = np.full(lat.shape,np.nan),np.full(lat.shape,np.nan),np.full(lat.shape,np.nan),np.full(lat.shape,np.nan)

    for ii,jj in itertools.product(range(n),range(m)):
        if (ii == 0) & (jj == 0): # upper left corner
            dx[ii,jj] = geodesic((lat[ii,jj],lon[ii,jj]),(lat[ii,jj+1],lon[ii,jj+1])).m
            dy[ii,jj] = geodesic((lat[ii,jj],lon[ii,jj]),(lat[ii+1,jj],lon[ii+1,jj])).m
            du_y[ii,jj] = u[ii+1,jj]-u[ii,jj]
            dv_x[ii,jj] = v[ii,jj+1]-v[ii,jj]
        elif (ii == 0) & (jj == (m-1)): # upper right corner
            dx[ii,jj] = geodesic((lat[ii,jj],lon[ii,jj]),(lat[ii,jj-1],lon[ii,jj-1])).m
            dy[ii,jj] = geodesic((lat[ii,jj],lon[ii,jj]),(lat[ii+1,jj],lon[ii+1,jj])).m
            du_y[ii,jj] = u[ii+1,jj]-u[ii,jj]
            dv_x[ii,jj] = v[ii,jj]-v[ii,jj-1]
        elif (ii == (n-1)) & (jj == 0): # lower left corner
            dx[ii,jj] = geodesic((lat[ii,jj],lon[ii,jj]),(lat[ii,jj+1],lon[ii,jj+1])).m
            dy[ii,jj] = geodesic((lat[ii,jj],lon[ii,jj]),(lat[ii-1,jj],lon[ii-1,jj])).m
            du_y[ii,jj] = u[ii,jj]-u[ii-1,jj]
            dv_x[ii,jj] = v[ii,jj+1]-v[ii,jj]
        elif (ii == (n-1)) & (jj == (m-1)): # lower right corner
            dx[ii,jj] = geodesic((lat[ii,jj],lon[ii,jj]),(lat[ii,jj-1],lon[ii,jj-1])).m
            dy[ii,jj] = geodesic((lat[ii,jj],lon[ii,jj]),(lat[ii-1,jj],lon[ii-1,jj])).m
            du_y[ii,jj] = u[ii,jj]-u[ii-1,jj]
            dv_x[ii,jj] = v[ii,jj]-v[ii,jj-1]
        elif ii == 0: # upper row interior cols
            dx[ii,jj] = geodesic((lat[ii,jj-1],lon[ii,jj-1]),(lat[ii,jj+1],lon[ii,jj+1])).m
            dy[ii,jj] = geodesic((lat[ii,jj],lon[ii,jj]),(lat[ii+1,jj],lon[ii+1,jj])).m
            du_y[ii,jj] = u[ii+1,jj]-u[ii,jj]
            dv_x[ii,jj] = v[ii,jj+1]-v[ii,jj-1]
        elif ii == (n-1): # lower row interior col s
            dx[ii,jj] = geodesic((lat[ii,jj-1],lon[ii,jj-1]),(lat[ii,jj+1],lon[ii,jj+1])).m
            dy[ii,jj] = geodesic((lat[ii,jj],lon[ii,jj]),(lat[ii-1,jj],lon[ii-1,jj])).m
            du_y[ii,jj] = u[ii,jj]-u[ii-1,jj]
            dv_x[ii,jj] = v[ii,jj+1]-v[ii,jj-1]
        elif jj == 0: # left column interior rows
            dx[ii,jj] = geodesic((lat[ii,jj],lon[ii,jj]),(lat[ii,jj+1],lon[ii,jj+1])).m
            dy[ii,jj] = geodesic((lat[ii-1,jj],lon[ii-1,jj]),(lat[ii+1,jj],lon[ii+1,jj])).m
            du_y[ii,jj] = u[ii+1,jj]-u[ii-1,jj]
            dv_x[ii,jj] = v[ii,jj+1]-v[ii,jj]
        elif jj == (m-1): # right column interior rows
            dx[ii,jj] = geodesic((lat[ii,jj],lon[ii,jj]),(lat[ii,jj-1],lon[ii,jj-1])).m
            dy[ii,jj] = geodesic((lat[ii-1,jj],lon[ii-1,jj]),(lat[ii+1,jj],lon[ii+1,jj])).m
            du_y[ii,jj] = u[ii+1,jj]-u[ii-1,jj]
            dv_x[ii,jj] = v[ii,jj]-v[ii,jj-1]
        else: # completely interior
            dx[ii,jj] = geodesic((lat[ii,jj-1],lon[ii,jj-1]),(lat[ii,jj+1],lon[ii,jj+1])).m 
            dy[ii,jj] = geodesic((lat[ii-1,jj],lon[ii-1,jj]),(lat[ii+1,jj],lon[ii+1,jj])).m
            du_y[ii,jj] = u[ii+1,jj]-u[ii-1,jj]
            dv_x[ii,jj] = v[ii,jj+1]-v[ii,jj-1]

    du_dy = du_y/dy
    dv_dx = dv_x/dx
    vort = dv_dx-du_dy
            
    return np.array(vort),np.array(du_dy), np.array(dv_dx)

#===============================================================================================================

def coriolis_par(lat):
    import numpy as np
    
    # lat is -90 to 90 degrees
    
    omega = omega = 7.29e-5
    f = 2. * omega * np.sin(np.deg2rad(lat))
    return f

#===============================================================================================================

def seasonal_clim(data):
    ''' must be for an xarray dataset'''
    
    import numpy as np

    # -------------------------------------------
    # weighted seasonal
    # -------------------------------------------

    # get months
    month_length = data.time.dt.days_in_month

    # calculate the weights by grouping by 'time.season'.
    weights = month_length.groupby('time.season') / month_length.groupby('time.season').sum()

    # calculate the weighted average
    seas_clim = (data * weights).groupby('time.season').sum(dim='time')   

    # set the places that are now zero from the weights to nans
    seas_clim = seas_clim.where(seas_clim != 0,np.nan)

    return seas_clim
    
#===============================================================================================================

def monthly_clim(data):
    ''' must be for an xarray dataset'''
    
    # -------------------------------------------
    # monthly
    # -------------------------------------------

    mon_clim = data.groupby('time.month').mean('time') 
    
    return mon_clim

#===============================================================================================================

def find_coast(arr):
    import itertools
    import xarray as xr
    import numpy as np
    
    # create empty lists to add row and col info to
    rowind = []
    colind = []

    # create an array of indices
    rows = np.arange(arr.shape[0])
    cols = np.arange(arr.shape[1])

    # find if the sum of a block around a point is a nan (meaning one of the values at least must have been a nan)
    for row,col in itertools.product(rows,cols):
#         cond = (~np.isnan(arr[row,col])) & (np.isnan(np.sum(arr[max(0,row-1):min(arr.shape[0],row+2),max(0,col-1):min(arr.shape[1],col+2)])))
        rowcond = (np.isnan(np.sum(arr[max(0,row-1):min(arr.shape[0],row+2),col])))
        colcond = (np.isnan(np.sum(arr[row,max(0,col-1):min(arr.shape[1],col+2)])))

#         if  (~np.isnan(arr[row,col])) & cond):
        if  (~np.isnan(arr[row,col])) & (rowcond | colcond):
            rowind.append(rows[row].tolist())
            colind.append(cols[col].tolist())
    
    return np.array(rowind), np.array(colind)

#===============================================================================================================
# mask coastlines ---------------------------------------------------------------------#
def mask_coast(inlat,inlon,inmask,mask_lat, mask_lon):
    import xarray as xr
    import numpy as np
    
    inlat = np.array(inlat)
    inlon = np.array(inlon)
    lat = np.array(mask_lat)
    lon = np.array(mask_lon)
    inmask = np.array(inmask)

    outmask=[]

    for lo,la in zip(inlon,inlat):

        if len(lon[lon<=lo])>0 and len(lat[lat>=la])>0 and len(lon[lon>=lo])>0 and len(lat[lat<=la])>0:
            lon_lim = [lon[lon<=lo][-1],lon[lon>=lo][0]]
            lat_lim = [lat[lat<=la][-1],lat[lat>=la][0]]

            mask_lon = (lon == lon_lim[0]) | (lon == lon_lim[1])
            mask_lat = (lat == lat_lim[0]) | (lat == lat_lim[1])

            mask_tmp = inmask[mask_lat,:]
            mask_tmp = mask_tmp[:,mask_lon]

            outmask.append(np.mean(mask_tmp)>0)
        else:
            outmask.append(False)

    outmask = np.array(outmask)

    return outmask

#===============================================================================================================
# mask coastlines ---------------------------------------------------------------------#

def mask_coast_roobaert_wide(c_lat,c_lon,bounds = cbounds):
    import xarray as xr
    import numpy as np
    
    data=xr.open_dataset('/tigress/GEOCLIM/LRGROUP/shared_data/pco2_flux_coastal_Roobaert/mask_ocean.nc')
    mask_coast=np.array(data.mask_coastal2).astype(int).T
    lat=np.array(data.latitude)
    lon=np.array(data.longitude)
    
    inlat = np.array(c_lat)
    inlon = np.array(c_lon)

    mask_lon=np.logical_and(lon>bounds[0],lon<bounds[1])
    mask_lat=np.logical_and(lat>bounds[2],lat<bounds[3])

    lon=lon[mask_lon]
    lat=lat[mask_lat]

    mask_coast=mask_coast[mask_lat]
    mask_coast=mask_coast[:,mask_lon]

    lonlon,latlat=np.meshgrid(lon,lat)

    # what are these for? I should probably fix it
    lon_dot=np.array([70,70])
    lat_dot=np.array([10,19.5])

    mask=[]
    for lo,la in zip(c_lon,c_lat):
        if len(lon[lon<=lo])>0 and len(lat[lat>=la])>0 and len(lon[lon>=lo])>0 and len(lat[lat<=la])>0:
            lon_lim=[lon[lon<=lo][-1],lon[lon>=lo][0]]
            lat_lim=[lat[lat<=la][-1],lat[lat>=la][0]]
            mask_lon=np.logical_or(lon==lon_lim[0],lon==lon_lim[1])
            mask_lat=np.logical_or(lat==lat_lim[0],lat==lat_lim[1])
            mask_tmp=mask_coast[mask_lat]
            mask_tmp=mask_tmp[:,mask_lon]
            mask.append(np.mean(mask_tmp)>0)
        else:
            mask.append(False)
    mask=np.array(mask)
    return mask

#===============================================================================================================

# mask coastlines ---------------------------------------------------------------------#
def mask_coast_roobaert_narrow(c_lat,c_lon,bounds = cbounds):
    import xarray as xr
    import numpy as np
    data=xr.open_dataset('/tigress/GEOCLIM/LRGROUP/shared_data/pco2_flux_coastal_Roobaert/mask_ocean.nc')
    mask_coast=np.array(data.mask_coastal1).astype(int).T
    lat=np.array(data.latitude)
    lon=np.array(data.longitude)

    mask_lon=np.logical_and(lon>bounds[0],lon<bounds[1])
    mask_lat=np.logical_and(lat>bounds[2],lat<bounds[3])

    lon=lon[mask_lon]
    lat=lat[mask_lat]

    mask_coast=mask_coast[mask_lat]
    mask_coast=mask_coast[:,mask_lon]

    lonlon,latlat=np.meshgrid(lon,lat)

    # what are these for? I should probably fix it
    lon_dot=np.array([70,70])
    lat_dot=np.array([10,19.5])

    mask=[]
    for lo,la in zip(c_lon,c_lat):
        if len(lon[lon<=lo])>0 and len(lat[lat>=la])>0 and len(lon[lon>=lo])>0 and len(lat[lat<=la])>0:
            lon_lim=[lon[lon<=lo][-1],lon[lon>=lo][0]]
            lat_lim=[lat[lat<=la][-1],lat[lat>=la][0]]
            mask_lon=np.logical_or(lon==lon_lim[0],lon==lon_lim[1])
            mask_lat=np.logical_or(lat==lat_lim[0],lat==lat_lim[1])
            mask_tmp=mask_coast[mask_lat]
            mask_tmp=mask_tmp[:,mask_lon]
            mask.append(np.mean(mask_tmp)>0)
        else:
            mask.append(False)
    mask=np.array(mask)
    return mask

#===============================================================================================================

def get_continuous_cmap(hex_list, float_list=None):
    import numpy as np
    import matplotlib.colors as mcolors
    
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map
        
        from here: https://towardsdatascience.com/beautiful-custom-colormaps-with-matplotlib-5bab3d1f0e72
        '''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

#===============================================================================================================

def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

#===============================================================================================================

def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]

#===============================================================================================================
def regrid_2_woa(invar,inlat,inlon):
    import numpy as np
    
    # read woa file to get grid
    ds_WOA = "xr.open_dataset(../data/woa_processed.nc)"
    
    xx,yy = np.meshgrid(inlon,inlat)
    xx = xx.flatten()
    yy = yy.flatten()

    xx_WOA,yy_WOA = np.meshgrid(ds_WOA.lon,ds_WOA.lat)

    invar_WOA = np.zeros((len(ds_WOA.lat),len(ds_WOA.lon)))*np.nan
    invar = np.array(invar)

    # find the lons and lats of TCD vals at this time

    points = np.array( (xx,yy) ).T
    temp_values = temp_var.flatten()

    invar_WOA = griddata(points, temp_values, (xx_WOA,yy_WOA) ,method='linear')

    return np.array(invar_WOA,dtype = float), np.array(ds_WOA.lat), np.array(ds_WOA.lon)

#===============================================================================================================

# binning for one variable ------------------------------------------------------------#
def latlonbin(invar,lat,lon,bounds = cbounds,binwidth = 0.25):
    import numpy as np
    import pandas as pd
    
    # create a pandas dataframe
    df = pd.DataFrame(dict(
            invar = np.array(invar),
            lat= np.array(lat),
            lon= np.array(lon),
        ))

    # create 1 degree bins
    latedges = np.arange(bounds[2]-(binwidth/2),bounds[3]+(binwidth/2),binwidth)
    lat_inds = list(range(len(latedges)-1))

    lonedges = np.arange(bounds[0]-(binwidth/2),bounds[1]+(binwidth/2),binwidth)
    lon_inds = list(range(len(lonedges)-1))

    latbins = latedges[1:]-(binwidth/2)
    lonbins = lonedges[1:]-(binwidth/2)

    df['latedges'] = pd.cut(lat, latedges)
    df['lonedges'] = pd.cut(lon, lonedges)
    df['latbins_ind'] = pd.cut(lat, latedges,labels = lat_inds)
    df['lonbins_ind'] = pd.cut(lon, lonedges,labels = lon_inds)
    df['lat_lon_indx']=df.groupby(['latbins_ind', 'lonbins_ind']).ngroup()
    grouped = df.groupby(['latbins_ind', 'lonbins_ind'])

    invar_BINNED = np.zeros((len(latbins),len(lonbins)), dtype=np.ndarray)
    invar_BINNED[:] = np.nan

    invar_binned_ave = np.zeros((len(latbins),len(lonbins)), dtype=np.ndarray)
    invar_binned_ave[:] = np.nan
    
    invar_bincounts = np.zeros((len(latbins),len(lonbins)), dtype=np.ndarray)
    invar_bincounts[:] = np.nan


    #extract the data for each group
    for name, group in grouped:
        i = np.array(group.latbins_ind)
        j = np.array(group.lonbins_ind)

        invar_BINNED[i[0],j[0]] = group.invar

        invar_binned_ave[i[0],j[0]] = np.nanmean(group.invar)   
        
        invar_bincounts[i[0],j[0]] = len(group.invar[np.isfinite(group.invar)]) 

    return np.array(invar_binned_ave,dtype = float),np.array(invar_bincounts,dtype = float),latbins,lonbins

#===============================================================================================================

# month of min doxy  --------------------------------------------------------------------#
def latlonbin_min_doxy(doxy,lat,lon,bounds = cbounds,binwidth = 0.25):
    import numpy as np
    import pandas as pd
    
    # create a pandas dataframe
    df = pd.DataFrame(dict(
            doxy = np.array(doxy),
            lat= np.array(lat),
            lon= np.array(lon),
            month = np.array(doxy.time.dt.month)
        ))
    
    # create 1 degree bins
    latedges = np.arange(bounds[2]-(binwidth/2),bounds[3]+(binwidth/2),binwidth)
    lat_inds = list(range(len(latedges)-1))

    lonedges = np.arange(bounds[0]-(binwidth/2),bounds[1]+(binwidth/2),binwidth)
    lon_inds = list(range(len(lonedges)-1))

    latbins = latedges[1:]-(binwidth/2)
    lonbins = lonedges[1:]-(binwidth/2)

    df['latedges'] = pd.cut(lat, latedges)
    df['lonedges'] = pd.cut(lon, lonedges)
    df['latbins_ind'] = pd.cut(lat, latedges,labels = lat_inds)
    df['lonbins_ind'] = pd.cut(lon, lonedges,labels = lon_inds)
    df['lat_lon_indx']=df.groupby(['latbins_ind', 'lonbins_ind']).ngroup()
    grouped = df.groupby(['latbins_ind', 'lonbins_ind'])

    min_doxy = np.zeros((len(latbins),len(lonbins)), dtype=np.ndarray) * np.nan
    min_doxy_mon = np.zeros((len(latbins),len(lonbins)), dtype=np.ndarray) * np.nan
    min_doxy_seas = np.zeros((len(latbins),len(lonbins)), dtype=np.ndarray) * np.nan


    #extract the data for each group
    for name, group in grouped:
        i = np.array(group.latbins_ind)
        j = np.array(group.lonbins_ind)
        gpmonth = np.array(group.month)
        gpdoxy = np.array(group.doxy)

        # find month of minimum OCD
        if ~np.isnan(np.nanmin(gpdoxy)):
            
            min_doxy[i[0],j[0]] = np.nanmin(gpdoxy)
            
            ind = np.nanargmin(gpdoxy)
            min_month = gpmonth[ind]
            if len(gpdoxy[gpdoxy == np.nanmin(gpdoxy)])>1:
                print('Duplicate Values in Minimum.', len(gpdoxy[gpdoxy == np.nanmin(gpdoxy)]),
                      gpmonth[gpdoxy == np.nanmin(gpdoxy)] )
            
            min_doxy_mon[i[0],j[0]] = min_month
            
            if (min_month == 3) | (min_month == 4) | (min_month == 5):
                min_doxy_seas[i[0],j[0]] = 0
            elif (min_month == 6) | (min_month == 7) | (min_month == 8):
                min_doxy_seas[i[0],j[0]] = 1
            elif (min_month == 9) | (min_month == 10) | (min_month == 11):
                min_doxy_seas[i[0],j[0]] = 2
            elif (min_month == 12) | (min_month == 1) | (min_month == 2):
                min_doxy_seas[i[0],j[0]] = 3
  

    return(np.array(min_doxy,dtype = float),np.array(min_doxy_mon,dtype = float),
           np.array(min_doxy_seas,dtype = float),
           latbins,lonbins)

#===============================================================================================================

# month of max doxy -----------------------------------------------------------------#
def latlonbin_max_doxy(doxy,lat,lon,bounds = cbounds,binwidth = 0.25):
    import numpy as np
    import pandas as pd
    
    # create a pandas dataframe
    df = pd.DataFrame(dict(
            doxy = np.array(doxy),
            lat= np.array(lat),
            lon= np.array(lon),
            month = np.array(doxy.time.dt.month)
        ))
    
    # create 1 degree bins
    latedges = np.arange(bounds[2]-(binwidth/2),bounds[3]+(binwidth/2),binwidth)
    lat_inds = list(range(len(latedges)-1))

    lonedges = np.arange(bounds[0]-(binwidth/2),bounds[1]+(binwidth/2),binwidth)
    lon_inds = list(range(len(lonedges)-1))

    latbins = latedges[1:]-(binwidth/2)
    lonbins = lonedges[1:]-(binwidth/2)

    df['latedges'] = pd.cut(lat, latedges)
    df['lonedges'] = pd.cut(lon, lonedges)
    df['latbins_ind'] = pd.cut(lat, latedges,labels = lat_inds)
    df['lonbins_ind'] = pd.cut(lon, lonedges,labels = lon_inds)
    df['lat_lon_indx']=df.groupby(['latbins_ind', 'lonbins_ind']).ngroup()
    grouped = df.groupby(['latbins_ind', 'lonbins_ind'])

    max_doxy = np.zeros((len(latbins),len(lonbins)), dtype=np.ndarray) * np.nan
    max_doxy_mon = np.zeros((len(latbins),len(lonbins)), dtype=np.ndarray) * np.nan
    max_doxy_seas = np.zeros((len(latbins),len(lonbins)), dtype=np.ndarray) * np.nan


    #extract the data for each group
    for name, group in grouped:
        i = np.array(group.latbins_ind)
        j = np.array(group.lonbins_ind)
        gpmonth = np.array(group.month)
        gpdoxy = np.array(group.doxy)

        # find month of minimum OCD
        if ~np.isnan(np.nanmax(gpdoxy)):
            
            max_doxy[i[0],j[0]] = np.nanmax(gpdoxy)
            
            ind = np.nanargmin(gpdoxy)
            max_month = gpmonth[ind]
            if len(gpdoxy[gpdoxy == np.nanmax(gpdoxy)])>1:
                print('Duplicate Values in Maximum.', len(gpdoxy[gpdoxy == np.nanmax(gpdoxy)]),
                      gpmonth[gpdoxy == np.nanmax(gpdoxy)] )
            
            max_doxy_mon[i[0],j[0]] = max_month
            
            if (max_month == 3) | (max_month == 4) | (max_month == 5):
                max_doxy_seas[i[0],j[0]] = 0
            elif (max_month == 6) | (max_month == 7) | (max_month == 8):
                max_doxy_seas[i[0],j[0]] = 1
            elif (max_month == 9) | (max_month == 10) | (max_month == 11):
                max_doxy_seas[i[0],j[0]] = 2
            elif (max_month == 12) | (max_month == 1) | (max_month == 2):
                max_doxy_seas[i[0],j[0]] = 3
  

    return(np.array(max_doxy,dtype = float),np.array(max_doxy_mon,dtype = float),
           np.array(max_doxy_seas,dtype = float),
           latbins,lonbins)

#===============================================================================================================

def IOD_year_group(invar,inlat,inlon,intime,begin,end,IODyears, region = 'none'):
    import numpy as np
    data= []
    lat = []
    lon = []
    time = []
    month = []
    season = []
    for ii,year in enumerate(IODyears):
        start_time = str(year) + begin
        end_time = str(year+1) + end
        time_slice = slice(start_time, end_time)
#         print(year)
        
        if region == 'wAS':
            data.extend(np.array(invar.sel(time_wAS=time_slice)))
            lat.extend(np.array(inlat.sel(time_wAS=time_slice)))
            lon.extend(np.array(inlon.sel(time_wAS=time_slice)))
            time.extend(np.array(intime.sel(time_wAS=time_slice)))
            t = intime.sel(time_wAS=time_slice)
            month.extend(np.array(t.dt.month))
            season.extend(np.array(t.dt.season))
        elif region == 'eAS':
            data.extend(np.array(invar.sel(time_eAS=time_slice)))
            lat.extend(np.array(inlat.sel(time_eAS=time_slice)))
            lon.extend(np.array(inlon.sel(time_eAS=time_slice)))
            time.extend(np.array(intime.sel(time_eAS=time_slice)))
            t = intime.sel(time_eAS=time_slice)
            month.extend(np.array(t.dt.month))
            season.extend(np.array(t.dt.season))
        elif region == 'wBoB':
            data.extend(np.array(invar.sel(time_wBoB=time_slice)))
            lat.extend(np.array(inlat.sel(time_wBoB=time_slice)))
            lon.extend(np.array(inlon.sel(time_wBoB=time_slice)))
            time.extend(np.array(intime.sel(time_wBoB=time_slice)))
            t = intime.sel(time_wBoB=time_slice)
            month.extend(np.array(t.dt.month))
            season.extend(np.array(t.dt.season))
        elif region == 'eBoB':
            data.extend(np.array(invar.sel(time_eBoB=time_slice)))
            lat.extend(np.array(inlat.sel(time_eBoB=time_slice)))
            lon.extend(np.array(inlon.sel(time_eBoB=time_slice)))
            time.extend(np.array(intime.sel(time_eBoB=time_slice)))
            t = intime.sel(time_eBoB=time_slice)
            month.extend(np.array(t.dt.month))
            season.extend(np.array(t.dt.season))
        elif region == 'none':
            data.extend(np.array(invar.sel(time=time_slice)))
            lat.extend(np.array(inlat.sel(time=time_slice)))
            lon.extend(np.array(inlon.sel(time=time_slice)))
            time.extend(np.array(intime.sel(time=time_slice)))
            t = intime.sel(time=time_slice)
            month.extend(np.array(t.dt.month))
            season.extend(np.array(t.dt.season))
        
        
    return np.array(data),np.array(lat),np.array(lon),np.array(time),np.array(month),np.array(season)

#===============================================================================================================

def IOD_year_group_grid(invar,begin,end,IODyears, roll = True):
    import numpy as np
    import xarray as xr
    
    data= []
    for ii,year in enumerate(IODyears):
        start_time = str(year) + begin
        end_time = str(year+1) + end
        time_slice = slice(start_time, end_time)
        data.append(invar.sel(time=time_slice))
        
    # add all the data together
    sp_data = xr.concat(data, dim='time')
    # take the mean for each month of all the years
    data = sp_data.groupby('time.month').mean(dim='time')
    #start in June instead of 01
    if roll == True:
        data = data.roll(month=-5,roll_coords = False)
    
    return data, sp_data

#===============================================================================================================

def IOD_year_group_area(invar,begin,end,IODyears,cbounds=[48.5, 102.5,-1.5, 33],coast = True):
    import numpy as np
    import xarray as xr
    
    data= []
    space_data = []
    for ii,year in enumerate(IODyears):
        start_time = str(year) + begin
        end_time = str(year+1) + end
        time_slice = slice(start_time, end_time)
        sp_data = invar.sel(time=time_slice)
        if coast == True:
            xx,yy = np.meshgrid(sp_data.lon,sp_data.lat)
            mask = mask_coast(xx.flatten(),yy.flatten(),cbounds)
            sp_data_c = sp_data.stack(allpoints=['lat','lon'])[:,mask]
            area_avg = np.mean(sp_data_c,1)
            
        else:
            area_avg = sp_data.stack(allpoints = ['lat','lon']).mean(dim='allpoints')
           
        data.append(area_avg)
        space_data.append(sp_data)
        
    # add all the data together
    data_concat = xr.concat(data, dim='time')
    # average
    # take the mean for each month of all the years
    data_mean = data_concat.groupby('time.month').mean(dim='time')
    #start in June instead of 01
    data_mean = data_mean.roll(month=-5,roll_coords = False)
        
    return data, data_mean, space_data

#===============================================================================================================

# correlation for TCD and OCD --------------------------------------------------------------#
def interannual_space_correlate(var1,var2,lat,lon,bounds = cbounds,binwidth=1):
    import numpy as np
    from scipy import stats 
    from tqdm import tqdm
    import itertools
    import pandas as pd
    
    # create a pandas dataframe
    df = pd.DataFrame(dict(
            var1=np.array(var1),
            var2=np.array(var2),
            lat=np.array(lat),
            lon=np.array(lon),
        ))

    # set to nans all the values where there isn't a tcd and ocd value
    ind = (df['var1'].isnull()) | (df['var2'].isnull()) 
    df.loc[(ind),'var1']=np.nan
    df.loc[(ind),'var2']=np.nan

    # create bins
    latedges = np.arange(bounds[2]-(binwidth/2),bounds[3]+(binwidth/2),binwidth)
    lat_inds = list(range(len(latedges)-1))

    lonedges = np.arange(bounds[0]-(binwidth/2),bounds[1]+(binwidth/2),binwidth)
    lon_inds = list(range(len(lonedges)-1))

    latbins = latedges[1:]-(binwidth/2)
    lonbins = lonedges[1:]-(binwidth/2)
    
    df['latedges'] = pd.cut(lat, latedges)
    df['lonedges'] = pd.cut(lon, lonedges)
    df['latbins_ind'] = pd.cut(lat, latedges,labels = lat_inds)
    df['lonbins_ind'] = pd.cut(lon, lonedges,labels = lon_inds)
    df['lat_lon_indx']=df.groupby(['latbins_ind', 'lonbins_ind']).ngroup()
    grouped = df.groupby(['latbins_ind', 'lonbins_ind'])

    SLOPE = np.zeros((len(latbins),len(lonbins)), dtype=np.ndarray)*np.nan
    INTERCEPT = np.zeros((len(latbins),len(lonbins)), dtype=np.ndarray)*np.nan
    R_VALUE = np.zeros((len(latbins),len(lonbins)), dtype=np.ndarray)*np.nan
    P_VALUE = np.zeros((len(latbins),len(lonbins)), dtype=np.ndarray)*np.nan
    STD_ERR = np.zeros((len(latbins),len(lonbins)), dtype=np.ndarray)*np.nan
    invar1_bincounts = np.zeros((len(latbins),len(lonbins)), dtype=np.ndarray)*np.nan
    invar2_bincounts = np.zeros((len(latbins),len(lonbins)), dtype=np.ndarray)*np.nan
    
    #extract the data for each group
    for name, group in grouped:
        i = np.array(group.latbins_ind)
        j = np.array(group.lonbins_ind)

        # find correlations if all in the group are not nans
        if ~((group.var1.isnull().all()) & (group.var2.isnull().all())) :
#             print(name)
            nanmask = ~np.isnan(group.var2) & ~np.isnan(group.var1)
            slope, intercept, r_value, p_value, std_err=stats.linregress(group.var1[nanmask],group.var2[nanmask])
            
            SLOPE[i[0],j[0]] = slope
            INTERCEPT[i[0],j[0]] = intercept
            R_VALUE[i[0],j[0]] = r_value
            P_VALUE[i[0],j[0]] = p_value
            STD_ERR[i[0],j[0]] = std_err
            invar1_bincounts[i[0],j[0]] = len(group.var1[np.isfinite(group.var1)]) 
            invar2_bincounts[i[0],j[0]] = len(group.var1[np.isfinite(group.var1)]) 

    return(np.array(SLOPE,dtype = float),np.array(INTERCEPT,dtype = float),np.array(R_VALUE,dtype = float), 
           np.array(P_VALUE,dtype = float),np.array(STD_ERR,dtype = float),latbins,lonbins,
           np.array(invar1_bincounts,dtype = float),np.array(invar2_bincounts,dtype = float))

#===============================================================================================================

def seasonal_space_correlate(invar1,invar2):
    import numpy as np
    from scipy import stats 
    from tqdm import tqdm
    import itertools
    
    SLOPE = np.full(invar1.shape[1:3],np.nan)
    INTERCEPT = np.full(invar1.shape[1:3],np.nan)
    R_VALUE = np.full(invar1.shape[1:3],np.nan)
    P_VALUE = np.full(invar1.shape[1:3],np.nan)
    STD_ERR = np.full(invar1.shape[1:3],np.nan)
    
    n = len(invar1.lat)
    m = len(invar1.lon)
    with tqdm(total=n * m) as pbar:
        for ii,jj in itertools.product(np.arange(len(invar1.lat)),np.arange(len(invar1.lon))):
            tempvar1 = invar1[:,ii,jj]
            tempvar2 = invar2[:,ii,jj]

            nanmask = ~np.isnan(tempvar1) & ~np.isnan(tempvar2)

            if (tempvar1[nanmask].size != 0) & (tempvar2[nanmask].size != 0):
                slope, intercept, r_value, p_value, std_err=stats.linregress(tempvar1[nanmask],tempvar2[nanmask])

                SLOPE[ii,jj] = slope
                INTERCEPT[ii,jj] = intercept
                R_VALUE[ii,jj] = r_value
                P_VALUE[ii,jj] = p_value
                STD_ERR[ii,jj] = std_err
                
            pbar.update(1)
        
    return SLOPE, INTERCEPT, R_VALUE, P_VALUE, STD_ERR

#===============================================================================================================

# create pcolormesh lat/lon vals ------------------------------------------------------#
def pcolor_xy(x,y):
    # extend longitude by 2
    x_extend = np.zeros(x.size+2)
    # fill in internal values
    x_extend[1:-1] = x # fill up with original values
    # fill in extra endpoints
    x_extend[0] = x[0]-np.diff(x)[0]
    x_extend[-1] = x[-1]+np.diff(x)[-1]
    # calculate the midpoints
    x_pcolormesh_midpoints = x_extend[:-1]+0.5*(np.diff(x_extend))

    # extend latitude by 2
    y_extend = np.zeros(y.size+2)
    # fill in internal values
    y_extend[1:-1] = y
    # fill in extra endpoints
    y_extend[0] = y[0]-np.diff(y)[0]
    y_extend[-1] = y[-1]+np.diff(y)[-1]
    # calculate the midpoints
    y_pcolormesh_midpoints = y_extend[:-1]+0.5*(np.diff(y_extend))
    
    return x_pcolormesh_midpoints,y_pcolormesh_midpoints

#===============================================================================================================

def plot_slice(ax,volume, orientation, index,cmap):
    
    orientation2slicefunc = {
        "depth" : lambda ar:ar[index,:,:], 
        "lat" : lambda ar:ar[:,index,:],  
        "lon" : lambda ar:ar[:,:,index]
    }
    volume_slice = orientation2slicefunc[orientation](volume)
    
    if orientation == 'depth':
        p =  ax.pcolormesh(volume.lat,volume.lon,volume_slice, cmap = cmap)
        ax.colorbar(p, ax=ax)
        plt.ylabel('lat')
        plt.xlabel('lon')
        
    elif orientation == 'lat':
        p = ax.pcolormesh(volume.lon,volume.depth,volume_slice, cmap = cmap)
        plt.colorbar(p, ax=ax)
        ax.invert_yaxis() # put zero in upper left corner instead of lower left
        plt.xlabel('lon')
        plt.ylabel('depth')
        
    elif orientation == 'lon':
        p = ax.pcolormesh(volume.lat,volume.depth,volume_slice,cmap = cmap)
        plt.colorbar(p, ax=ax)
        ax.invert_yaxis() # put zero in upper left corner instead of lower left
        plt.xlabel('lat')
        plt.ylabel('depth')

#===============================================================================================================
        
def plot_slice_CD(ax,volume, orientation, index):
    orientation2slicefunc = {
        "lat" : lambda ar:ar[index,:],  
        "lon" : lambda ar:ar[:,index]
    }
    volume_slice = orientation2slicefunc[orientation](volume)

    if orientation == 'lat':
        ax.plot(volume.lon,volume_slice, color = 'k', linewidth = 3)

        
    elif orientation == 'lon':
        ax.plot(volume.lat,volume_slice, color = 'k', linewidth = 3)

#===============================================================================================================

def plot_map_inset(ax,volume, orientation,index):
    
    if orientation == 'lat':
        add_land(ax)
        plt.plot(volume.lon,np.ones(volume.lon.shape)*np.array(volume.lat[index]), color = 'w')
    elif orientation == 'lon':
        add_land(ax)
        plt.plot(np.ones(volume.lat.shape)*np.array(volume.lon[index]),volume.lat, color = 'w')  
        
#===============================================================================================================

def plot_mean_slice(ax,volume, orientation,lat_ind_begin,lat_ind_end,lon_ind_begin,lon_ind_end,cmap):
    
    orientation2slicefunc = {
        "lat" : lambda ar:ar[:,lat_ind_begin:lat_ind_end+1,lon_ind_begin:lon_ind_end+1],  
        "lon" : lambda ar:ar[:,lat_ind_begin:lat_ind_end+1,lon_ind_begin:lon_ind_end+1]
    }
    volume_slice = orientation2slicefunc[orientation](volume)
        
    if orientation == 'lat':
        temp_lon = volume.lon[lon_ind_begin:lon_ind_end+1]
        p = ax.pcolormesh(temp_lon,volume.depth,np.nanmean(volume_slice,1), cmap = cmap)
        plt.colorbar(p, ax=ax)
        ax.invert_yaxis() # put zero in upper left corner instead of lower left
        ax.set_xlabel('lon')
        ax.set_ylabel('depth')
        
    elif orientation == 'lon':
        temp_lat = volume.lat[lat_ind_begin:lat_ind_end+1]
        p = ax.pcolormesh(temp_lat,volume.depth,np.nanmean(volume_slice,2),cmap = cmap)
        plt.colorbar(p, ax=ax)
        ax.invert_yaxis() # put zero in upper left corner instead of lower left
        plt.xlabel('lat')
        plt.ylabel('depth')
        
#===============================================================================================================

def plot_mean_slice_CD(ax,volume, orientation,lat_ind_begin,lat_ind_end,lon_ind_begin,lon_ind_end):
    orientation2slicefunc = {
        "lat" : lambda ar:ar[lat_ind_begin:lat_ind_end+1,lon_ind_begin:lon_ind_end+1],  
        "lon" : lambda ar:ar[lat_ind_begin:lat_ind_end+1,lon_ind_begin:lon_ind_end+1]
    }
    volume_slice = orientation2slicefunc[orientation](volume)
    print(volume_slice.shape)

    if orientation == 'lat':
        temp_lon = volume.lon[lon_ind_begin:lon_ind_end+1]
        ax.plot(temp_lon,np.nanmean(volume_slice,0), color = 'k', linewidth = 3)

        
    elif orientation == 'lon':
        temp_lat = volume.lat[lat_ind_begin:lat_ind_end+1]
        ax.plot(temp_lat,np.nanmean(volume_slice,1), color = 'k', linewidth = 3)
        
#===============================================================================================================

def plot_mean_map_inset(ax,volume, orientation,lat_ind_begin,lat_ind_end,lon_ind_begin,lon_ind_end):
    
    add_land(ax)
    box = [float(volume.lon[lon_ind_begin]),float(volume.lon[lon_ind_end]),
           float(volume.lat[lat_ind_begin]),float(volume.lat[lat_ind_end])]
    print(box)
    p = Rectangle(
        (box[0], box[2]), box[1]-box[0], box[3]-box[2],
        linewidth=2,fill=False,color='white')

    ax.add_patch(p)

    
#===============================================================================================================
# filtering a 2D matrix ---------------------------------------------------------------#
def gaus_filter_nan(img,sigma = 1):
    # import required packages
    from astropy.convolution import convolve
    from astropy.convolution import Gaussian2DKernel
    
    # create a kernal with std = 1
    kernel = Gaussian2DKernel(x_stddev=1)
    
    #convolve that with the 2d matrix
    img_conv = convolve(img, kernel)

    # set original nan points back to nans
    img_conv[np.isnan(img)] = np.nan

    return img_conv    
        
#===============================================================================================================

def find_coast(arr):
    import itertools
    import numpy as np
    
    # create empty lists to add row and col info to
    rowind = []
    colind = []

    # create an array of indices
    rows = np.arange(arr.shape[0])
    cols = np.arange(arr.shape[1])

    # find if the sum of a block around a point is a nan (meaning one of the values at least must have been a nan)
    for row,col in itertools.product(rows,cols):

        rowcond = (np.isnan(np.sum(arr[max(0,row-1):min(arr.shape[0],row+2),col])))
        colcond = (np.isnan(np.sum(arr[row,max(0,col-1):min(arr.shape[1],col+2)])))

        if  (~np.isnan(arr[row,col])) & (rowcond | colcond):
            rowind.append(rows[row].tolist())
            colind.append(cols[col].tolist())
    
    return np.array(rowind), np.array(colind)      
        
 
#===============================================================================================================       
        
def order_coast(loninds,latinds,sta_zero):
    import numpy as np
    
    # find based on radius
    zipped_lists = zip(loninds, latinds)
    sorted_pairs = sorted(zipped_lists, reverse=True)

    # sort by lon
    tuples = zip(*sorted_pairs)
    lon_list,lat_list  = [ list(tuple) for tuple in  tuples]

    pos = []
    curr_sta = []
    rem_sta = sorted_pairs
    for i in range(len(lon_list)):
#         print(i)
        if i == 0:
            curr_sta.append(sta_zero)
            rem_sta.remove(sta_zero) 
            prev_sta = sta_zero
        else:
            prev_sta = curr_sta[i-1]

        start_len = len(curr_sta)
        for j,(lo, la) in enumerate(rem_sta):
            next_sta = (rem_sta[j])
            
            diff = tuple(map(lambda l, k: l - k, curr_sta[i], next_sta))
            
            # check uplr first
            if(next_sta != prev_sta) & (all(np.abs(diff) == [0,1])) | (all(np.abs(diff) == [1,0])):
                curr_sta.append(next_sta)
                rem_sta.remove(next_sta) 
                break

            # then check diagonals
            elif (next_sta != prev_sta) & (all(np.abs(diff) == [1,1])):
                curr_sta.append(next_sta)
                rem_sta.remove(next_sta) 
                break
                
        if len(curr_sta) == start_len:
            print('No Next Station Found. Returning Previous Stations Only.')
            print(curr_sta[i],rem_sta)
            break
        
    sta_lonind, sta_latind  = map(np.array, zip(*curr_sta))
    
    return sta_lonind, sta_latind
        
        
        