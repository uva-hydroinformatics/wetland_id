# -*- coding: utf-8 -*-
"""
Modules call GRASS GIS scripts to perform A* conditioning and TWI calculation, 
IRA conditioning, and DTW least-cost path analysis.

Created on Tue Apr 17 17:49:24 2018
@author: Gina O'Neil
"""

import subprocess
import os
from osgeo import gdal, gdal_array
import numpy as np
import scipy
from scipy import stats
import wetland_id_defaults as default
import sys
from scipy import ndimage
from raster_array_funcs import *
import grass.script as g
import grass.script.setup as gsetup
import shutil


##This is modified from geonet script but here we use grassgis r.hydrodem which implements Lind.&Creed 2005 IRA
def grass_At(dem_in):
    grass7bin = default.set_grassbin()
    startcmd = [grass7bin, '--config', 'path']
    p = subprocess.Popen(startcmd, shell=True, \
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        print "ERROR: Cannot find GRASS GIS 7 " \
              "start script (%s)" % startcmd
        sys.exit(-1)
    gisbase = out.strip('\n\r')
    
    if sys.platform.startswith('win'):
        gisdbdir = os.path.join(os.path.expanduser("~"), r"Documents\grassdata")
    else:
        gisdbdir = os.path.join(os.path.expanduser("~"), "grassdata")
    
    """Setting GRASS GIS env variables"""
    os.environ['GISBASE'] = gisbase
    os.environ['PATH'] += os.pathsep + os.path.join(gisbase, 'extrabin')
    home = os.path.expanduser("~")
    os.environ['PATH'] += os.pathsep + os.path.join(home, '.grass7', 'addons', 'scripts')
    os.environ['GISDBASE'] = gisdbdir
    gpydir = os.path.join(gisbase, "etc", "python")
    sys.path.append(gpydir)
  
    
	"""Create necessary GRASS GIS location/mapset"""   
    geotiff = dem_in #this should the be entire path/to/dem.tif that will be processed by Grass (filtered)
    locationGeonet = default.location
    mapsetGeonet = default.mapset
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
    
    tmpfile = dem_in.split('\\')[-1]
    geotiffmapraster = tmpfile[:-4] # this reads something like dem
    print 'GRASSGIS layer name: ',geotiffmapraster
    g.run_command('r.in.gdal', input=geotiff, \
                  output=geotiffmapraster,overwrite=True)
#    gtf = Parameters.geotransform
    #Flow computation for massive grids (float version)
	
	# Save the outputs as TIFs
    outputFAC_filename = geotiffmapraster + '_grass_at_fac.tif'
    outputFDR_filename = geotiffmapraster + '_grass_at_fdr.tif'
    outputTWI_filename = geotiffmapraster + '_grass_at_twi.tif'
    
    twi_at = os.path.join(default.roi_vars, outputTWI_filename)

    if not os.path.exists(twi_at):
	
        print "Calling the r.watershed command from GRASS GIS"
        subbasinThreshold = 1500
    
    ##uncommend to use swap memory options
    #    if xDemSize == 4000 or yDemSize == 4000:
    #        print ('using swap memory option for large size DEM')
    #        g.run_command('r.watershed',flags ='am',overwrite=True,\
    #						  elevation=geotiffmapraster, \
    #						  threshold=subbasinThreshold, \
    #						  drainage = 'dra1v23', memory = 30000)
    #
    #        g.run_command('r.watershed',flags ='am',overwrite=True,\
    #						  elevation=geotiffmapraster, \
    #						  threshold=subbasinThreshold, \
    #						  tci='at_twi', memory = 30000)
    #        
    #        g.run_command('r.watershed',flags ='a',overwrite=True,\
    #						  elevation=geotiffmapraster, \
    #						  threshold=subbasinThreshold, \
    #						  accumulation='acc1v23', memory = 30000)
    #        
    #    else : ....all below
        
        g.run_command('r.watershed',flags ='ab',overwrite=True,\
    						  elevation=geotiffmapraster, \
    						  threshold=subbasinThreshold, \
    						  accumulation='acc1v23',\
    						  drainage = 'dra1v23', \
                              tci = 'at_twi')
    		 
        # Save the outputs as TIFs
    
        g.run_command('r.out.gdal',overwrite=True,\
    					  input='acc1v23', type='Float64',\
    					  output=os.path.join(default.roi_vars,
    										  outputFAC_filename), format='GTiff')
        
        g.run_command('r.out.gdal',overwrite=True,\
    					  input = "dra1v23", type='Float32',\
    					  output=os.path.join(default.roi_vars,
    										  outputFDR_filename),\
    					  format='GTiff')
    
        g.run_command('r.out.gdal',overwrite=True,\
    					  input = "at_twi", type='Float64',\
    					  output=os.path.join(default.roi_vars,
    										  outputTWI_filename),\
    					  format='GTiff')    
    return twi_at             
         
def grass_IRA(dem_in):
    #'breach' refers to IRA
    tmpfile = dem_in.split('\\')[-1]
    geotiffmapraster = tmpfile[:-4]
    br_name = geotiffmapraster + '_grass_breach.tif'
    breached = os.path.join(default.roi_dems, br_name)
    
    if not os.path.exists(breached):    
        #https://grass.osgeo.org/grass74/manuals/addons/r.hydrodem.html
        grass7bin = default.set_grassbin()
        startcmd = [grass7bin, '--config', 'path']
        p = subprocess.Popen(startcmd, shell=True, \
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode != 0:
            print "ERROR: Cannot find GRASS GIS 7 " \
                  "start script (%s)" % startcmd
            sys.exit(-1)
        gisbase = out.strip('\n\r')
        
        if sys.platform.startswith('win'):
            gisdbdir = os.path.join(os.path.expanduser("~"), r"Documents\grassdata")
        else:
            gisdbdir = os.path.join(os.path.expanduser("~"), "grassdata")
        
        """Setting GRASS GIS env variables"""
        os.environ['GISBASE'] = gisbase
        os.environ['PATH'] += os.pathsep + os.path.join(gisbase, 'extrabin')
        home = os.path.expanduser("~")
        os.environ['PATH'] += os.pathsep + os.path.join(home, '.grass7', 'addons', 'scripts')
        os.environ['GISDBASE'] = gisdbdir
        gpydir = os.path.join(gisbase, "etc", "python")
        sys.path.append(gpydir)
        
    	"""Create necessary GRASS GIS location/mapset"""   
        geotiff = dem_in #this should the be entire path/to/dem.tif that will be processed by Grass (filtered)
        locationGeonet = default.location
        mapsetGeonet = default.mapset
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
        
        tmpfile = dem_in.split('\\')[-1]
        geotiffmapraster = tmpfile[:-4] # this reads something like dem
        print 'GRASSGIS layer name: ',geotiffmapraster
        g.run_command('r.in.gdal', input=geotiff, \
                      output=geotiffmapraster,overwrite=True)    
        
        print "Calling the r.hydrodem command from GRASS GIS to generate IRA DEM"
    
    #according to documentation, flag "a" applies the IRA parroach (lindsay & Creed (2005))\
    #and makes the output a depressionless DEM suitable for r.terraflow
        g.run_command('r.hydrodem',flags ='a',overwrite=True,\
    						  input=geotiffmapraster, \
    						  output='breach')        
    
    # Save the outputs as TIFs
    
        g.run_command('r.out.gdal',overwrite=True,\
    					  input='breach', type='Float64',\
    					  output= breached, format='GTiff')
    return breached

def grass_ira_twi(dem_in):
    #https://grass.osgeo.org/grass74/manuals/addons/r.hydrodem.html
    grass7bin = default.set_grassbin()
    startcmd = [grass7bin, '--config', 'path']
    p = subprocess.Popen(startcmd, shell=True,\
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    print p.returncode
    if p.returncode != 0:
        print "ERROR: Cannot find GRASS GIS 7 " \
              "start script (%s)" % startcmd
        sys.exit(-1)
    gisbase = out.strip('\n\r')
    
    if sys.platform.startswith('win'):
        gisdbdir = os.path.join(os.path.expanduser("~"), r"Documents\grassdata")
    else:
        gisdbdir = os.path.join(os.path.expanduser("~"), "grassdata")
    
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
    geotiff = dem_in #this should the be entire path/to/dem.tif that will be processed by Grass (filtered)
    locationGeonet = default.location
    mapsetGeonet = default.mapset
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
    
    tmpfile = dem_in.split('\\')[-1]
    geotiffmapraster = tmpfile[:-4] # this reads something like dem
    print 'GRASSGIS layer name: ',geotiffmapraster
    g.run_command('r.in.gdal', input=geotiff, \
                  output=geotiffmapraster,overwrite=True, mem=30000)

    #Flow computation for massive grids (float version)
	
    fdr_name = geotiffmapraster + '_grass_breach_fdr1.tif'
    acc_name = geotiffmapraster + '_grass_breach_acc.tif'
    twi_name = geotiffmapraster + '_grass_breach_twi.tif'
    stats_name = os.path.join(default.roi_log, (geotiffmapraster+"_stats.out"))
    ira_twi = os.path.join(default.roi_vars, twi_name) 
    
    if not os.path.exists(ira_twi):
    
        print "Calling the r.terraflow command from GRASS GIS"
        
        g.run_command('r.terraflow', \
                      flags='', overwrite=True,\
                      elevation=geotiffmapraster,\
                      direction='fdr', \
                      accumulation='acc',\
                      tci='iratwi', stats=stats_name, swatershed='swat', filled='flooded', mem=30000)
        
    #https://grass.osgeo.org/grass74/manuals/r.terraflow.html 		 
        # Save the outputs as TIFs

        g.run_command('r.out.gdal',overwrite=True,\
    					  input='fdr', type='Float64',\
    					  output=os.path.join(default.roi_vars, fdr_name), format='GTiff')
        
        g.run_command('r.out.gdal', overwrite=True,\
    					  input = "acc", type='Float64',\
    					  output=os.path.join(default.roi_vars, acc_name), format='GTiff')
    
        g.run_command('r.out.gdal', overwrite=True,\
    					  input = "iratwi", type='Float64',\
    					  output = ira_twi, format='GTiff')
        
        g.run_command('r.out.gdal', overwrite=True,\
    					  input = "floodend", type='Float64',\
    					  output = os.path.join(default.roi_dems, 'grassflooded.tif'), format='GTiff')
        
    return ira_twi


def grass_cost(cost_tif, source_shp):   
 
    grass7bin = default.set_grassbin()
    startcmd = [grass7bin, '--config', 'path']
    p = subprocess.Popen(startcmd, shell=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        print "ERROR: Cannot find GRASS GIS 7 " \
              "start script (%s)" % startcmd
        sys.exit(-1)
    gisbase = out.strip('\n\r')
    
    if sys.platform.startswith('win'):
        gisdbdir = os.path.join(os.path.expanduser("~"), r"Documents\grassdata")
    else:
        gisdbdir = os.path.join(os.path.expanduser("~"), "grassdata")
    
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
    geotiff = cost_tif #this should the be entire path/to/dem.tif that will be processed by Grass
    shp_vect = source_shp
    
    locationGeonet = default.location
    mapsetGeonet = default.mapset
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
    
    tmpfile = cost_tif.split('\\')[-1]
    geotiffmapraster = tmpfile[:-4] # this reads something like dem
    
    tmpfile = source_shp.split('\\')[-1]
    shp_vect_map = tmpfile[:-4]
    
    print 'GRASSGIS layer name: ', geotiffmapraster
    g.run_command('r.in.gdal', input=geotiff, \
                  output=geotiffmapraster,overwrite=True)
    
    g.run_command('v.in.ogr', input=shp_vect, \
                  output=shp_vect_map, overwrite=True)

    dtw_fname = os.path.join(default.roi_vars, geotiffmapraster + default.dtw_suf)
    dir_fname = os.path.join(default.roi_vars, geotiffmapraster + "_dir.tif")
    
    if not os.path.exists(dtw_fname):
        print "Calling the r.cost command from GRASS GIS"
    
        g.run_command('r.cost',flags ='k',overwrite=True, input=geotiffmapraster,\
                      start_points = shp_vect_map, output='dtw', outdir = "dtw_dir")
    
    		 # Save the outputs as TIFs
        g.run_command('r.out.gdal',overwrite=True,\
    					  input='dtw', type='Float64',\
    					  output= dtw_fname, format='GTiff')
    
        g.run_command('r.out.gdal',overwrite=True,\
    					  input='dtw_dir', type='Float32',\
    					  output=dir_fname, format='GTiff')
   
    return dtw_fname, dir_fname
