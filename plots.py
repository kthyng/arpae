'''
Plots for project.
'''

import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean.cm as cmo
import os
import shapely


land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m')
pc = cartopy.crs.PlateCarree()
merc = cartopy.crs.Mercator(central_longitude=-85.0)


def roi(sinks=None):
  '''Region of interest of possible starting locations for drifters

    Must be in regions with typical salinity < 33, 1000m depth, US waters
    '''


    # EEZ Polygon
    fname = 'data/useez/useez.shp'
    records = cartopy.io.shapereader.Reader(fname)
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



    extent = [-98, -80, 18, 31]

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

    # add sinks
    if sinks is not None:
        ax.plot(sinks[:,0], sinks[:,1], 'bo', markersize=10, transform=pc)


    os.makedirs('figures', exist_ok=True)
    fig.savefig('figures/roi.png', bbox_inches='tight')
    fig.savefig('figures/roi_lowres.png', bbox_inches='tight', dpi=70)
