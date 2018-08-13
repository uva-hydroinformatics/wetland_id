import os
import sys
import shutil
import subprocess
from time import clock
from pygeonet_rasterio import *
import grass.script as g
import grass.script.setup as gsetup

"""EDITED SCRIPT: removed some repetive commmands, removed swap memory option, added command to set GRASS GIS env vars,
move gsetup.init command earlier, separately installed r.stream.basins. for GRASS (see wetland ID README)"""

def grass(filteredDemArray):
    grass7bin = Parameters.set_grassbin()
    startcmd = [grass7bin, '--config', 'path']
    p = subprocess.Popen(startcmd, shell=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        print >>sys.stderr, "ERROR: Cannot find GRASS GIS 7 " \
              "start script (%s)" % startcmd
        sys.exit(-1)
    gisbase = out.strip('\n\r')
#    gisdb = os.path.join(os.path.expanduser("~"), "grassdata") #this does not direct anywhere, isn't used later in script?
    mswin = sys.platform.startswith('win')
    if mswin:
        gisdbdir = os.path.join(os.path.expanduser("~"), r"Documents\grassdata")
    else:
        gisdbdir = os.path.join(os.path.expanduser("~"), "grassdata")
    originalGeotiff = os.path.join(Parameters.demDataFilePath, Parameters.demFileName)
    
    """Setting GRASS GIS env variables"""
    os.environ['GISBASE'] = gisbase
    os.environ['PATH'] += os.pathsep + os.path.join(gisbase, 'extrabin')
    home = os.path.expanduser("~")
    os.environ['PATH'] += os.pathsep + os.path.join(home, '.grass7', 'addons', 'scripts')
    os.environ['GISDBASE'] = gisdbdir
    gpydir = os.path.join(gisbase, "etc", "python")
    sys.path.append(gpydir)
    if sys.platform.startswith('win'):
        gisdb = os.path.join(os.getenv('APPDATA', 'grassdata'))
    else:
        gisdb = os.path.join(os.getenv('HOME', 'grassdata'))
    
	"""Create necessary GRASS GIS location/mapset"""   
    geotiff = Parameters.pmGrassGISfileName
    locationGeonet = Parameters.location
    mapsetGeonet = Parameters.mapset
    grassGISlocation = os.path.join(gisdbdir, locationGeonet)
    if os.path.exists(grassGISlocation):
        print "Cleaning existing Grass location"
        shutil.rmtree(grassGISlocation)
    
#	must create PERMANENT mapset first
    gsetup.init(gisbase, gisdbdir, locationGeonet, 'PERMANENT')
    
    print 'Making the geonet location'
    g.run_command('g.proj', georef=geotiff, location = locationGeonet)
    print 'Existing Mapsets after making locations:'
    g.read_command('g.mapsets', flags = 'l')
    print 'Setting GRASSGIS environ'
    print 'Making mapset now'
    g.run_command('g.mapset', flags = 'c', mapset = mapsetGeonet,\
                  location = locationGeonet, dbase = gisdbdir)
   
    # Read the filtered DEM
    print 'Import filtered DEM into GRASSGIS and '\
          'name the new layer with the DEM name'
    tmpfile = Parameters.demFileName # this reads something like skunk.tif
    geotiffmapraster = tmpfile.split('.')[0]
    print 'GRASSGIS layer name: ',geotiffmapraster
    g.run_command('r.in.gdal', input=geotiff, \
                  output=geotiffmapraster,overwrite=True)
#    gtf = Parameters.geotransform
    #Flow computation for massive grids (float version)
	
	# Save the outputs as TIFs
    outlet_filename = geotiffmapraster + '_outlets.tif'
    outputFAC_filename = geotiffmapraster + '_fac.tif'
    outputFDR_filename = geotiffmapraster + '_fdr.tif'
    outputBAS_filename = geotiffmapraster + '_basins.tif'
	
    print "Calling the r.watershed command from GRASS GIS"
    subbasinThreshold = defaults.thresholdAreaSubBasinIndexing

#    if (not hasattr(Parameters, 'xDemSize')) or (not hasattr(Parameters, 'yDemSize')):
#        Parameters.yDemSize=np.size(filteredDemArray,0)
#        Parameters.xDemSize=np.size(filteredDemArray,1)
#     #if broken, remove 'm' from 'am' 
#    if Parameters.xDemSize > 4000 or Parameters.yDemSize > 4000:
#        print ('using swap memory option for large size DEM')
#        g.run_command('r.watershed',flags ='am',overwrite=True,\
#						  elevation=geotiffmapraster, \
#						  threshold=subbasinThreshold, \
#						  drainage = 'dra1v23', memory = 20000)
#        g.run_command('r.watershed',flags ='am',overwrite=True,\
#						  elevation=geotiffmapraster, \
#						  threshold=subbasinThreshold, \
#						  accumulation='acc1v23', memory = 20000)
#    else :
    g.run_command('r.watershed',flags ='a',overwrite=True,\
						  elevation=geotiffmapraster, \
						  threshold=subbasinThreshold, \
						  accumulation='acc1v23',\
						  drainage = 'dra1v23')
    print 'Identify outlets by negative flow direction'
    g.run_command('r.mapcalc',overwrite=True,\
					  expression='outletmap = if(dra1v23 >= 0,null(),1)')
    print 'Convert outlet raster to vector'
    g.run_command('r.to.vect',overwrite=True,\
					  input = 'outletmap', output = 'outletsmapvec',\
					  type='point')
    print 'Delineate basins according to outlets'
    g.run_command('r.stream.basins',overwrite=True,\
					  direction='dra1v23',points='outletsmapvec',\
					  basins = 'outletbains', memory = 20000)
		 # Save the outputs as TIFs
    g.run_command('r.out.gdal',overwrite=True,\
					  input='outletmap', type='Float32',\
					  output=os.path.join(Parameters.geonetResultsDir,
										  outlet_filename),\
					  format='GTiff')
    g.run_command('r.out.gdal',overwrite=True,\
					  input='acc1v23', type='Float64',\
					  output=os.path.join(Parameters.geonetResultsDir,
										  outputFAC_filename),\
					  format='GTiff')
    g.run_command('r.out.gdal',overwrite=True,\
					  input = "dra1v23", type='Float64',\
					  output=os.path.join(Parameters.geonetResultsDir,
										  outputFDR_filename),\
					  format='GTiff')
    g.run_command('r.out.gdal',overwrite=True,\
					  input = "outletbains", type='Int16',\
					  output=os.path.join(Parameters.geonetResultsDir,
										  outputBAS_filename),\
					  format='GTiff')
		
def main():
    print Parameters.pmGrassGISfileName
    filteredDemArray = read_geotif_filteredDEM()
    grass(filteredDemArray)

if __name__ == '__main__':
    t0 = clock()
    main()
    t1 = clock()
    print "time taken to complete flow accumulation:", t1-t0, " seconds"
