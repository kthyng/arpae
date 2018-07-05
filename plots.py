'''
Plots for project.
'''

import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean.cm as cmo
import os
import shapely
import shapely.ops
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import netCDF4 as netCDF
import tracpy
import tracpy.plotting
from glob import glob
import fiona


land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m')
pc = cartopy.crs.PlateCarree()
merc = cartopy.crs.Mercator(central_longitude=-85.0)
extent = [-98, -80, 18, 31]

os.makedirs('figures', exist_ok=True)

def roi(grid, sinks=None, sinkarrows=None, seeds=None):
    '''Region of interest of possible starting locations for drifters

    Must be in regions with typical salinity < 33, 1000m depth, US waters
    '''

    fname = 'figures/roi'

    # EEZ Polygon
    dname = 'data/useez/useez.shp'
    records = cartopy.io.shapereader.Reader(dname)
    lines = []
    for record, geometry in zip(records.records(), records.geometries()):
        line = shapely.ops.linemerge(geometry)
        if isinstance(line, shapely.geometry.linestring.LineString):
            lines.append(line)

    # multi_line = shapely.geometry.MultiLineString(lines)
    eezlines = shapely.ops.linemerge(lines)
    # eez[5] happens to have the gulf boundary
    x, y = eezlines[5].xy
    # buffer fixed self-intersection problem
    # https://stackoverflow.com/questions/20833344/fix-invalid-polygon-python-shapely
    eez = shapely.geometry.Polygon(list(zip(x, y))).buffer(0)

    fig = plt.figure(figsize=(8,6))# (9.4, 7.7))
    ax = fig.add_subplot(111, projection=merc)
    # ax = fig.add_axes([0.06, 0.01, 0.93, 0.95], projection=merc)
    ax.set_extent(extent, pc)
    gl = ax.gridlines(linewidth=0.2, color='gray', alpha=0.5, linestyle='-', draw_labels=True)
    # the following two make the labels look like lat/lon format
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = mpl.ticker.FixedLocator(np.arange(-100, -70, 2))
    gl.ylocator = mpl.ticker.FixedLocator(np.arange(10, 31, 2))
    gl.xlabels_bottom = False  # turn off labels where you don't want them
    gl.ylabels_right = False
    # add background land
    ax.add_feature(land_50m, facecolor='0.8')

    # plot 1000m isobath
    cs = ax.contour(grid.lon_rho, grid.lat_rho, grid.h, [1000], colors='purple', transform=pc, linewidths=0.7)
    # https://gis.stackexchange.com/questions/99917/converting-matplotlib-contour-objects-to-shapely-objects
    p = cs.collections[0].get_paths()[1]
    v = p.vertices
    x, y = v[:,0], v[:,1]
    # bathymetry polygon for 1000m isobath
    bathy = shapely.geometry.Polygon([(i[0], i[1]) for i in zip(x,y)])
    # ax.add_geometries([bathy], pc, facecolor='none', edgecolor='0.1', linewidth=0.7)


    ax.add_geometries([eez], pc, facecolor='none', edgecolor='darkcyan')

    # find seaweed paddock region of interest (intersection)
    seaweed = bathy.intersection(eez)
    ax.add_geometries([seaweed], pc, facecolor='magenta', alpha=0.5, edgecolor='m')

    # save seaweed paddock polygon
    # Define a polygon feature geometry with one attribute
    schema = {
        'geometry': 'Polygon',
        'properties': {'id': 'int'},
    }

    # Write a new Shapefile
    os.makedirs('calcs/seaweed', exist_ok=True)
    with fiona.open('calcs/seaweed/seaweed.shp', 'w', 'ESRI Shapefile', schema) as c:
        ## If there are multiple geometries, put the "for" loop here
        c.write({
            'geometry': shapely.geometry.mapping(seaweed),
            'properties': {'id': 123},
        })


    # add sinks
    if sinks is not None:
        ax.plot(sinks.T[0], sinks.T[1], 'rx', markersize=10, transform=pc)
        fname += '_sinklocs'

    # add sink arrows
    if sinkarrows is not None:
        iu, jv = sinkarrows
        ax.quiver(grid.lon_u[::20,::20], grid.lat_u[::20,::20], iu[::20,::20], jv[::20,::20], transform=pc)
        fname += '_lon0_%2.2f_lat0_%2.2f_sinkarrows' % (abs(sinks[0]), sinks[1])

    # add initial drifter locations
    if seeds is not None:
        lon0, lat0 = seeds
        ax.plot(lon0, lat0, 'g.', transform=pc)
        fname += '_seeds'

    fig.savefig(fname + '.png', bbox_inches='tight')
    fig.savefig(fname + '_lowres.png', bbox_inches='tight', dpi=70)
    plt.close(fig)


def sinklocs():
    '''plot map with sinklocs.'''

    dates = glob('tracks/*')  # pull out dates
    dates = [date.split('/')[1] for date in dates]
    sinks = glob('tracks/%s/*_s_0.01gc.nc' % dates[0])  # pull out sink locations
    sinklocs = [(-float(sink.split('/')[-1].split('_')[1]), float(sink.split('/')[-1].split('_')[3])) for sink in sinks]

    colors = ['r','orange','g', 'b']

    fig = plt.figure(figsize=(8,6))# (9.4, 7.7))
    ax = fig.add_subplot(111, projection=merc)
    # ax = fig.add_axes([0.06, 0.01, 0.93, 0.95], projection=merc)
    ax.set_extent(extent, pc)
    gl = ax.gridlines(linewidth=0.2, color='gray', alpha=0.5, linestyle='-', draw_labels=True)
    # the following two make the labels look like lat/lon format
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = mpl.ticker.FixedLocator(np.arange(-100, -70, 2))
    gl.ylocator = mpl.ticker.FixedLocator(np.arange(10, 31, 2))
    gl.xlabels_bottom = False  # turn off labels where you don't want them
    gl.ylabels_right = False
    # add background land
    ax.add_feature(land_50m, facecolor='0.8')

    # add seaweed polygon
    polyname = 'calcs/seaweed/seaweed.shp'
    ax.add_geometries(cartopy.io.shapereader.Reader(polyname).geometries(),
                      pc, edgecolor='darkcyan', facecolor='none')

    for sink, color in zip(sinklocs, colors):
        ax.plot(sink[0], sink[1], 'o', color=color, markersize=10, transform=pc)
    fig.savefig('figures/sinklocs.png', bbox_inches='tight')
    fig.savefig('figures/sinklocs_lowres.png', bbox_inches='tight', dpi=70)


def results(dates):
    '''Plot overall results.'''

    loc = 'http://terrebonne.tamu.edu:8080/thredds/dodsC/NcML/gom_roms_hycom'
    grid_file = 'gom03_grd_N050_new.nc'
    proj = tracpy.tools.make_proj('nwgom-pyproj')
    grid = tracpy.inout.readgrid(grid_file, proj=proj)

    dd = 50  # to decimate drifters by
    speeds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    colors = ['r', 'orange', 'y', 'g', 'b', 'purple', 'brown']
    sinklocs = np.array([[-95, 26.75], [-91, 26.75], [-88, 27.5], [-85.5, 25.5]])
    lon0, lat0 = abs(sinklocs[0])

    fig = plt.figure(figsize=(8,6))# (9.4, 7.7))
    ax = fig.add_subplot(111, projection=merc)
    # ax = fig.add_axes([0.06, 0.01, 0.93, 0.95], projection=merc)
    ax.set_extent(extent, pc)
    gl = ax.gridlines(linewidth=0.2, color='gray', alpha=0.5, linestyle='-', draw_labels=True)
    # the following two make the labels look like lat/lon format
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = mpl.ticker.FixedLocator(np.arange(-100, -70, 2))
    gl.ylocator = mpl.ticker.FixedLocator(np.arange(10, 31, 2))
    gl.xlabels_bottom = False  # turn off labels where you don't want them
    gl.ylabels_right = False
    # add background land
    ax.add_feature(land_50m, facecolor='0.8')

    # overlay seaweed paddock polygon

    for date in dates:
        for speed, color in zip(speeds, colors):
            fname = 'lon0_%2.2f_lat0_%2.2f_s_%2.2f' % (lon0, lat0, speed)
            d = netCDF.Dataset('tracks/%s/%sgc.nc' % (date, fname))
            xg = d['xg'][:]; yg = d['yg'][:]
            xg = xg[::dd,:]; yg = yg[::dd,:]
            ind = xg == -1
            lonp, latp, _ = tracpy.tools.interpolate2d(xg, yg, grid, 'm_ij2ll')
            lonp[ind] = np.nan; latp[ind] = np.nan

            # plot tracks
            ax.plot(lonp.T, latp.T, color=color, alpha=0.2, lw=0.2, transform=pc)


def hist():
    '''Plot histogram of locations using tracpy.plotting.'''


    loc = 'http://terrebonne.tamu.edu:8080/thredds/dodsC/NcML/gom_roms_hycom'
    grid_file = 'gom03_grd_N050_new.nc'
    proj = tracpy.tools.make_proj('nwgom-pyproj')
    grid = tracpy.inout.readgrid(grid_file, proj=proj)

    # read in tracks
    dates = glob('tracks/*')  # pull out dates
    dates = [date.split('/')[1] for date in dates]
    sinks = glob('tracks/%s/*_s_0.01gc.nc' % dates[0])  # pull out sink locations
    sinklocs = [(-float(sink.split('/')[-1].split('_')[1]), float(sink.split('/')[-1].split('_')[3])) for sink in sinks]
    speeds = glob('tracks/%s/lon0_%2.2f_lat0_%2.2f_s_*gc.nc' % (dates[0], -sinklocs[0][0], sinklocs[0][1]))
    speeds = [float(speed[-9:-5]) for speed in speeds]

    for sinkloc in sinklocs:
        for speed in speeds:
            Files = glob('tracks/*/lon0_%2.2f_lat0_%2.2f_s_%2.2fgc.nc' % (-sinkloc[0], sinkloc[1], speed))

            bins = (100,100)
            fname = 'calcs/hist/lon0_%2.2f_lat0_%2.2f_s_%2.2f.npz' % (-sinkloc[0], sinkloc[1], speed)
            if not os.path.exists(fname):
                H = np.zeros(bins)
                for File in Files:
                    d = netCDF.Dataset(File)
                    xg = d['xg'][:]; yg = d['yg'][:]
                    # convert to projected coordinates
                    xpt, ypt, _ = tracpy.tools.interpolate2d(xg, yg, grid, 'm_ij2xy')
                    Ht, xedges, yedges = np.histogram2d(xpt.flatten(), ypt.flatten(),
                                                       range=[[grid.x_rho.min(),
                                                               grid.x_rho.max()],
                                                              [grid.y_rho.min(),
                                                               grid.y_rho.max()]],
                                                       bins=bins)
                    H += Ht
                    d.close()


                os.makedirs('calcs/hist', exist_ok=True)
                np.savez(fname, H=H, xedges=xedges, yedges=yedges)
            else:
                d = np.load(fname)
                H = d['H']; xedges = d['xedges']; yedges = d['yedges']
                d.close()


            fig, ax = tracpy.plotting.background(grid, extent=extent,
                                       outline=[0,0,0,0], figsize=(10,8),
                                       hlevs=np.hstack(([10, 20], np.arange(50, 500, 50), np.arange(500, 5000, 500))),
                                       halpha=0.5)
            tracpy.plotting.hist(None, None, proj, 'test', grid, tind='all', which='pcolor',
                                 cmap=cmo.amp, cbcoords=[0.45, 0.225, 0.4, 0.02],
                                 fig=fig, ax=ax, bins=(100, 100), N=100, xlims=None,
                                 ylims=None, C=None, Title=None, weights=None, H=H,
                                 Label='Drifter locations (%)', binscale=None,
                                 xedges=xedges, yedges=yedges, logscale=True,
                                 crsproj=cartopy.crs.LambertConformal(central_latitude= 30, central_longitude=-94))

            # add seaweed polygon
            polyname = 'calcs/seaweed/seaweed.shp'
            ax.add_geometries(cartopy.io.shapereader.Reader(polyname).geometries(),
                              pc, edgecolor='darkcyan', facecolor='none')

            os.makedirs('figures/hist', exist_ok=True)
            fig.savefig('figures/hist/%s.png' % fname.split('/')[-1][:-4], bbox_inches='tight')
            fig.savefig('figures/hist/%s_lowres.png' % fname.split('/')[-1][:-4], bbox_inches='tight', dpi=70)
            plt.close(fig)


def spread():
    '''Calculate and plot max spread from sinkloc.'''

    # loc = 'http://terrebonne.tamu.edu:8080/thredds/dodsC/NcML/gom_roms_hycom'
    grid_file = 'gom03_grd_N050_new.nc'
    proj = tracpy.tools.make_proj('nwgom-pyproj')
    grid = tracpy.inout.readgrid(grid_file, proj=proj)

    # get parameters
    dates = glob('tracks/*')  # pull out dates
    dates = [date.split('/')[1] for date in dates]
    sinks = glob('tracks/%s/*_s_0.01gc.nc' % dates[0])  # pull out sink locations
    sinklocs = [(-float(sink.split('/')[-1].split('_')[1]), float(sink.split('/')[-1].split('_')[3])) for sink in sinks]
    speeds = glob('tracks/%s/lon0_%2.2f_lat0_%2.2f_s_*gc.nc' % (dates[0], -sinklocs[0][0], sinklocs[0][1]))
    speeds = [float(speed[-9:-5]) for speed in speeds]

    fig1, axes1 = plt.subplots(2,4, sharex=True, sharey=True, figsize=(10,8))
    fig2, axes2 = plt.subplots(2,4, sharex=True, sharey=True, figsize=(10,8))
    fig3, axes3 = plt.subplots(2,4, sharex=True, sharey=True, figsize=(10,8))
    fig4, axes4 = plt.subplots(2,4, sharex=True, sharey=True, figsize=(10,8))
    colors = ['r','orange','g', 'b']
    # alphas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    markers = ['*', 'o', 'x', '+', '.', '^','s']
    for i, (speed, marker) in enumerate(zip(speeds, markers)):
        splot = []; meanplot = []; stdplot = []; maxplot = []
        dateplot = []; sinkloclon = []; sinkloclat = []
        for sinkloc, color in zip(sinklocs[3:], colors[3:]):
            meanplot2 = []; maxplot2 = [];
            for date in dates:
                fname = 'calcs/spread/%s/lon0_%2.2f_lat0_%2.2f_s_%2.2f.npz' % (date, -sinkloc[0], sinkloc[1], speed)

                if not os.path.exists(fname):
                    File = 'tracks/%s/lon0_%2.2f_lat0_%2.2f_s_%2.2fgc.nc' % (date, -sinkloc[0], sinkloc[1], speed)
                    d = netCDF.Dataset(File)
                    xg = d['xg'][:]; yg = d['yg'][:]
                    # convert to projected coordinates
                    xpt, ypt, _ = tracpy.tools.interpolate2d(xg, yg, grid, 'm_ij2xy')

                    sinklocx, sinklocy = proj(sinkloc[0], sinkloc[1])

                    # calculate distance of tracks from sinkloc
                    dist = np.sqrt( (xpt - sinklocx)**2 + (ypt - sinklocy)**2)

                    os.makedirs('calcs/spread/%s' % date, exist_ok=True)
                    np.savez(fname, t=d['tp'][0,:], mean=np.nanmean(dist, axis=0),
                             max=np.nanmax(dist, axis=0), std=np.nanstd(dist, axis=0))

                else:
                    d = np.load(fname)
                    t = d['t']
                    dmean = d['mean']; dmax = d['max']; dstd = d['std']
                    d.close()
                # ax.plot(speed, dmean.max()/1000, color=color, alpha=0.5, marker=marker)
                splot.append(speed)
                dateplot.append(date)
                sinkloclon.append(sinkloc[0])
                sinkloclat.append(sinkloc[1])
                meanplot.append(dmean.max()/1000)
                meanplot2.append(dmean.max()/1000)
                stdplot.append(dstd)
                maxplot.append(dmax.max()/1000)
                maxplot2.append(dmax.max()/1000)

            ax = axes2.flatten()[i]
            ax.hist(meanplot2, range=(0,200), bins=20, color=color, label='lon0_%2.2f_lat0_%2.2f'%(-sinkloc[0], sinkloc[1]))
            ax.set_title('speed=%2.2f m/s\n=%2.2f kts' % (speed*(12/5),speed*(12/5)*1.94384))
            if i in [4,5,6]:
                ax.set_xlabel('distance from sinkloc [km]')
            ax.set_ylim(0,50)

            ax = axes3.flatten()[i]
            ax.hist(maxplot2, range=(0,200), bins=20, color=color, label='lon0_%2.2f_lat0_%2.2f'%(-sinkloc[0], sinkloc[1]))
            ax.set_title('speed=%2.2f m/s\n=%2.2f kts' % (speed*(12/5),speed*(12/5)*1.94384))
            if i in [4,5,6]:
                ax.set_xlabel('distance from sinkloc [km]')
            ax.set_ylim(0,50)

        ax = axes1.flatten()[i]
        ax.hist(meanplot, range=(0,200), bins=20)#, label='speed=%2.2f'%speed)
        ax.set_title('speed=%2.2f m/s\n=%2.2f kts' % (speed*(12/5),speed*(12/5)*1.94384))
        if i in [4,5,6]:
            ax.set_xlabel('distance from sinkloc [km]')
        ax.set_ylim(0,170)

        ax = axes4.flatten()[i]
        ax.hist(maxplot, range=(0,200), bins=20)#, label='speed=%2.2f'%speed)
        ax.set_title('speed=%2.2f m/s\n=%2.2f kts' % (speed*(12/5),speed*(12/5)*1.94384))
        if i in [4,5,6]:
            ax.set_xlabel('distance from sinkloc [km]')
        ax.set_ylim(0,170)

        os.makedirs('figures/spread', exist_ok=True)
        fig1.savefig('figures/spread/maxofmean.png', bbox_inches='tight')
        fig1.savefig('figures/spread/maxofmean_lowres.png', bbox_inches='tight', dpi=70)
        fig2.savefig('figures/spread/maxofmean_locs.png', bbox_inches='tight')
        fig2.savefig('figures/spread/maxofmean_locs_lowres.png', bbox_inches='tight', dpi=70)
        fig3.savefig('figures/spread/maxofmax_locs.png', bbox_inches='tight')
        fig3.savefig('figures/spread/maxofmax_locs_lowres.png', bbox_inches='tight', dpi=70)
        fig4.savefig('figures/spread/maxofmax.png', bbox_inches='tight')
        fig4.savefig('figures/spread/maxofmax_lowres.png', bbox_inches='tight', dpi=70)

if __name__ == "__main__":
    spread()
