#! /usr/bin/env python
import os
import shutil
import sys
import gdal
import wetland_id_defaults as default

"""
Folder structure for pyGeoNet is as follows
geoNetHomeDir : defines where files will be written
e.g.
geoNetHomeDir = "C:\\Mystuff\\IO_Data\\"
        --- \\data     (input lidar files will be read from this folder)
        --- \\results  (outputs from pygeonet will be written to this folder)
        --- \\basinTiffs (intermediate GRASS GIS files will be written
                          and deleted from this location. some times these
                          files could be huge, so have enough space)

pmGrassGISfileName -- this is an important intermediate GRASS GIS file name.
# Skfmm parameters
numBasinsElements = 6

# Some used demFileNames
#ikawa_roi1_nutm54_clipped
#dem_2012_mission_v1

#PLEASE DO NOT CHANGE VARIABLES,UNLESS YOU KNOW WHAT YOU ARE DOING

"""
# Prepare GeoNet parameters just prior to main code execution
currentWorkingDir = os.getcwd()
geoNetHomeDir = r"D:\2ndStudy_ONeil\Scripts\wetland_identification"

demFileName = default.input_dem

demDataFilePath = default.roi_dems


geonetResultsDir = default.roi_geonet
geonetResultsBasinDir = default.roi_geonet

#channelheadFileName = "channelhead.shp"
channelheadFileName = "Hou_weights.tif"
channeljunctionFileName = "junction.shp"


gdal.UseExceptions()
ds = gdal.Open(os.path.join(demDataFilePath, demFileName), gdal.GA_ReadOnly)
driver = ds.GetDriver()
geotransform = ds.GetGeoTransform()
ary = ds.GetRasterBand(1).ReadAsArray()
demPixelScale = float(geotransform[1])
xLowerLeftCoord = float(geotransform[0])
yLowerLeftCoord = float(geotransform[3])
inputwktInfo = ds.GetProjection()


#GRASS GIS parameters
#set grass7bin variable based on OS, enter correct path to grass72.bat file, or add path to file to PATH
def set_grassbin():
	if sys.platform.startswith('win'):
			# MS Windows
			grass7bin = r'C:\Program Files\GRASS GIS 7.2.2\grass72.bat'
			# uncomment when using standalone WinGRASS installer
			# grass7bin = r'C:\Program Files (x86)\GRASS GIS 7.2.0\grass72.bat'
			# this can be avoided if GRASS executable is added to PATH
	elif sys.platform == 'darwin':
			# Mac OS X
			# TODO: this have to be checked, maybe unix way is good enough
			grass7bin = '/Applications/GRASS/GRASS-7.2.app/'
	return grass7bin

location = default.location
mapset = default.mapset

	
# Write shapefile file paths
shapefilepath = default.roi_geonet
driverName = 'ESRI Shapefile'

pointshapefileName = demFileName[:-4]+"_channelHeads"
pointFileName = os.path.join(shapefilepath, pointshapefileName+".shp")

drainagelinefileName = demFileName[:-4]+"_channelNetwork"
drainagelineFileName = os.path.join(shapefilepath, drainagelinefileName+".shp")

junctionshapefileName = demFileName[:-4]+"_channelJunctions"
junctionFileName = os.path.join(shapefilepath, junctionshapefileName+".shp")

streamcellFileName = os.path.join(geonetResultsDir,
                                  demFileName[:-4]+"_streamcell.csv")

xsshapefileName = demFileName[:-4]+"_crossSections"
xsFileName = os.path.join(shapefilepath, xsshapefileName+".shp")

banklinefileName = demFileName[:-4]+"_bankLines"
banklineFileName = os.path.join(shapefilepath, banklinefileName+".shp")


"""Things to be changed"""
# PM Filtered DEM to be used in GRASS GIS for flow accumulation
pmGrassGISfileName = os.path.join(geonetResultsDir, "PM_filtered_grassgis.tif")
#pmGrassGISfileName = os.path.join(demDataFilePath,demFileName)

# Skfmm parameters
numBasinsElements = 2


## Clean up previous results and recreate output folders
#if os.path.exists(geonetResultsBasinDir):
#    print "Cleaning old basinTiffs"
#    shutil.rmtree(geonetResultsBasinDir)
#
#if os.path.exists(geonetResultsDir):
#    print "Cleaning old results"
#    shutil.rmtree(geonetResultsDir)
###
#print "Making basinTiffs"
#os.mkdir(geonetResultsBasinDir)
####
#print "Making results"
#if not os.path.exists(geonetResultsDir):
#    os.mkdir(geonetResultsDir)
    