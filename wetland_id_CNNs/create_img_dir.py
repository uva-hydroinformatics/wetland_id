"""
Creates directory of images to be used as training and testing files for semantic segmentation using
convolutional neural networks.

Code reads in an entire project area geotiff and corresponding ground truth shapefile and splits
the project area into X by X sized tiles, and for each new tile the ground truth shapefile is clipped
to the same extents. If wetlands exist within the extents, additional geotiffs are created for each
wetland instance where wetland pixels are True and background pixels are False. No geotiff is created
if there are no wetlands in the tile.

Author: Gina O'Neil
Initial Version: Sept. 21, 2018
"""

from osgeo import gdal, ogr
import os
import sys
import subprocess
import glob


def create_imgs(tif_in, shp_in, img_dir):

    """
    tif_in = path to composite geotiff - raster contains input var info
    shp_in = path to ground truth shapefile
    img_dir = location for image directory
    """

    #open geotiff and get info
    dset = gdal.Open(tif_in)
    ext = dset.GetGeoTransform()
    width = dset.RasterXSize
    height = dset.RasterYSize

    tilesize = 500

    #first image #
    img_n = 0

    #cut composite.tif into tilesize x tilesize tiles
    for i in range(0, width, tilesize):
        for j in range(0, height, tilesize):
            #create subdir for each tile
            subdir = os.path.join(img_dir, "{}".format(str(img_n)))
            os.mkdir(subdir)
            w = min(i+tilesize, width) - i
            h = min(j+tilesize, height) - j
            #clip geotiff to extents and save as "IMG#.tif" in subdir
            cmd = "gdal_translate -of GTIFF -srcwin "+str(i)+", "+str(j)+", "+str(w)+", " \
                +str(h)+" " + tif_in + " " + subdir + r"\%d.tif" %(img_n)
            os.system(cmd)
            #increase img #
            img_n +=1

    #get list of new image subdirs
    img_subdirs = next(os.walk(img_dir))[1]

    #clip shp to each tile extents
    for x in range(len(img_subdirs)):
        tile_dir = os.path.join(img_dir, img_subdirs[x])
        tile_in = glob.glob(str(tile_dir)+r"\*.tif")[0] #should just be 1 tiff

        #open tile.tif
        dset = gdal.Open(tile_in)
        ulx, xres, xskew, uly, yskew, yres  = dset.GetGeoTransform()
        sizeX = dset.RasterXSize * xres
        sizeY = dset.RasterYSize * yres
        lrx = ulx + sizeX
        lry = uly + sizeY
        # format the extent coords
        extent = '{0} {1} {2} {3}'.format(ulx, lry, lrx, uly)
        dset = None

        #shp named with tile number in corresponding directory
        shp_out = tile_in[:-4]+ "_wetland.shp"

        # make clip command with ogr2ogr - default to shapefile format
        cmd = 'ogr2ogr ' + shp_out + ' ' + shp_in + ' -clipsrc ' + extent
        # call the command
        subprocess.call(cmd, shell=True)

        #open new shapefile to rasterize
        #TODO: delete shapefile after creation
        driver = ogr.GetDriverByName('ESRI Shapefile')
        dataSource = driver.Open(shp_out, 1) #make writable
        layer = dataSource.GetLayer()
        feature_count = layer.GetFeatureCount()

        #check for wetland instanc in tile extents
        if feature_count > 0:
            defn = layer.GetLayerDefn()
            field_count=defn.GetFieldCount()
            instance = ogr.FieldDefn("instance", ogr.OFTInteger)
            layer.CreateField(instance)
            feature = layer.GetNextFeature()

            #rasterize each wetland instance
            z = 0
            while feature is not None:
                feature.SetField("instance", z)
                layer.SetFeature(feature)

                #name new wetland mask geotiff as TILE#_wetland_WETLAND#.tif
                tif_out = shp_out[:-4] + "_%d.tif" % (z)

                cmd = 'gdal_rasterize -burn 1 -where \"instance={:d}\" -a_nodata 0 -ot Int32 -tr {} {} \"{}\" \"{}\"' \
                    .format(z, xres, yres, shp_out, tif_out)
                subprocess.Popen(cmd) #Popen kept double quotes needed for cmd, call did not
                feature.Destroy()
                feature = layer.GetNextFeature()
                z += 1
            feature = None

        #if no wetlands delete shapefile and do not create tiff
        else:
            driver.DeleteDataSource(shp_out)

        dataSource.Destroy()
    print("Image directory created with {} images".format(len(img_subdirs)))


tif_in = r"D:\3rdStudy_ONeil\data\s1_scratch\comp.tif"

shp_in = r"D:\2ndStudy_ONeil\Tool_testing\data\Site1\wetlands.shp"

img_dir = r"D:\3rdStudy_ONeil\data\s1_scratch\imgs"

create_imgs(tif_in, shp_in, img_dir)