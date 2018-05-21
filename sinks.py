'''
Create static velocity field that will push drifters
toward input origin.
'''

import tracpy
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('calcs/sinks', exist_ok=True)


def create(lon0, lat0, s):
    '''Create sinks file for use in tracpy simulation.

    Input sink location lon0, lat0, and speed s (m/s).'''

    fname = 'calcs/sinks/lon0_%2.2f_lat0_%2.2f_s_%2.2f.npz' % (abs(lon0), lat0, s)

    if os.path.exists(fname):
        d = np.load(fname)
        iu = d['iu']; jv = d['jv']; s = d['s'];
        lon0 = d['lon0']; lat0 = d['lat0']

    else:

        # create a unit vector pointing toward lon0, lat0
        # for every grid point in projected coordinates, then multiply by s
        # need to do this for u grid and v grid

        # read in grid
        loc = 'http://terrebonne.tamu.edu:8080/thredds/dodsC/NcML/gom_roms_hycom'
        grid_file = 'gom03_grd_N050_new.nc'
        proj = tracpy.tools.make_proj('nwgom-pyproj')
        grid = tracpy.inout.readgrid(grid_file, proj=proj)

        # convert sink to projected coordinates
        x0, y0 = proj(lon0, lat0)

        # calculate unit vector from grid nodes in projected coordinates
        # square root of squares of x distance from sink and y distance from sink
        # divided by magnitude to make it unit
        # negative is to make it a sink
        ## u grid ##
        xvec = -(grid.x_u - x0)  # x component
        yvec = -(grid.y_u - y0)  # y component
        mag = np.sqrt(xvec**2 + yvec**2)  # magnitude
        iu = xvec/mag  # x unit vector
        ju = yvec/mag  # y unit vector

        ## v grid ##
        xvec = -(grid.x_v - x0)  # x component
        yvec = -(grid.y_v - y0)  # y component
        mag = np.sqrt(xvec**2 + yvec**2)  # magnitude
        iv = xvec/mag  # x unit vector
        jv = yvec/mag  # y unit vector

        # # mask vectors
        # iu = np.ma.masked_where(grid.mask_u == 0, iu)
        # ju = np.ma.masked_where(grid.mask_u == 0, ju)
        # iv = np.ma.masked_where(grid.mask_v == 0, iv)
        # jv = np.ma.masked_where(grid.mask_v == 0, jv)

        # Check results by plotting arrows on map
        plt.quiver(grid.lon_u[::20,::20], grid.lat_u[::20,::20], iu[::20,::20], ju[::20,::20])
        plt.plot(lon0, lat0, 'ro')

        # multiply vectors by speed
        iu *= s
        ju *= s
        iv *= s
        jv *= s

        # save as npz file to be lazy
        np.savez(fname, iu=iu, jv=jv, s=s, lon0=lon0, lat0=lat0)

    return iu, jv
