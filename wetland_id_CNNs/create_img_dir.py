"""
Creates directory of images to be used as training and testing files for semantic segmentation using
convolutional neural networks.

Code reads in an entire project area geotiff and corresponding ground truth shapefile and splits
the project area into X by X sized tiles, and for each new tile the ground truth shapefile is clipped
to the same extents. If wetlands exist within the extents, additional geotiffs are created for each
wetland instance where wetland pixels are True and background pixels are False, while maintaining no data areas.
If there are no wetlands in the tile, no wetland geotiff is created and the image tile is moved to another directory.

Author: Gina O'Neil
Initial Version: Sept. 21, 2018
"""

from osgeo import gdal, ogr
import os
import sys
import subprocess
import glob
import shutil
import numpy as np
import raster_array_funcspy35 as ra
import shlex


def create_imgs(tif_in, shp_in, img_dir):

    """
    tif_in = path to composite geotiff - raster contains input var info
    shp_in = path to ground truth shapefile
    img_dir = location for image directory
    """

    #open geotiff and get info
    dset, meta = ra.geotif_to_array(tif_in, 'float')
    ext = meta['ext']

    width = meta['ncol']
    height = meta['nrow']

    tilesize = 640 #should be multiple of 64

    #first image #
    img_n = 0

    #cut composite.tif into tilesize x tilesize tiles
    for i in range(0, width, tilesize):
        for j in range(0, height, tilesize):
            #create subdir for each tile
            subdir = os.path.join(img_dir, "%02d" %(img_n))
            if os.path.exists(subdir):
                shutil.rmtree(subdir)
            os.mkdir(subdir)
            w = min(i+tilesize, width) - i
            h = min(j+tilesize, height) - j
            #clip geotiff to extents and save as "[IMG#].tif" in subdir
            cmd = "gdal_translate -of GTIFF -srcwin "+str(i)+", "+str(j)+", "+str(w)+", " \
                +str(h)+" " + tif_in + " " + subdir + r"\%02d.tif" %(img_n)

            os.system(cmd)
            #increase img #
            img_n +=1

    #get list of new image subdirs
    img_subdirs = next(os.walk(img_dir))[1]

    #clip shp to each tile extents
    for x in range(len(img_subdirs)):
        tile_dir = os.path.join(img_dir, img_subdirs[x])
        tile_in = glob.glob(str(tile_dir) + r"\*.tif")[0] #should just be 1 tiff

        #open tile.tif, nodata = -9999
        dset_tile, meta = ra.geotif_to_array(tile_in, 'float')

        #get tile info
        ulx, xres, xskew, uly, yskew, yres  = meta['ext']
        sizeX = meta['ncol'] * xres
        sizeY = meta['nrow'] * yres
        lrx = ulx + sizeX
        lry = uly + sizeY

        # format the extent coords
        extent = '{0} {1} {2} {3}'.format(ulx, lry, lrx, uly)

        #shp named with tile number in corresponding directory
        shp_out = tile_in[:-4]+ "-wetland.shp"

        # make clip command with ogr2ogr - default to shapefile format
        cmd = 'ogr2ogr ' + shp_out + ' ' + shp_in + ' -clipsrc ' + extent

        # call the command
        subprocess.call(cmd, shell=True)

        #read new shapefile in as temp array using MEM gdal driver
        driver = ogr.GetDriverByName('ESRI Shapefile')
        source_ds = driver.Open(shp_out, 1)
        source_layer = source_ds.GetLayer()
        #check if shapefile has wetlands
        feature_count = source_layer.GetFeatureCount()

        if feature_count > 0:

            print("Found {} wetlands in {}".format(feature_count, tile_in))

            #create a new attribute field to keep track of wetland instances
            defn = source_layer.GetLayerDefn()
            instance = ogr.FieldDefn("instance", ogr.OFTInteger)
            source_layer.CreateField(instance)

            source_srs = source_layer.GetSpatialRef()
            x_min, x_max, y_min, y_max = ulx, lrx, lry, uly #match extent to input img tile
            pixel_size = float(meta['pix_res'])

            NoData_value = 0 #actually non wetland (BG) value

            #create geotiff for each wetland instance, y
            for y in range(feature_count):
                tif_out = shp_out[:-4] + "-%d.tif" % (y)
                feature = source_layer.GetFeature(y)
                feature.SetField("instance", y)
                source_layer.SetFeature(feature)
                source_layer.SetAttributeFilter("instance={}".format(y)) #select wetland instance y

                # Create the destination data source
                target_ds = gdal.GetDriverByName('MEM').Create('', int(sizeX), int(sizeY*-1), gdal.GDT_Byte)
                target_ds.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
                band = target_ds.GetRasterBand(1)
                band.SetNoDataValue(NoData_value)
                band.FlushCache()

                # Rasterize
                gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[1], options=['ALL_TOUCHED=TRUE'])

                # Read as array
                temp_arr = band.ReadAsArray()

                #Mask tiff now has 1 where wetland instance exists and 0 elsewhere, designate actual no data areas
                temp_arr[dset_tile[:, :, 0] == -9999] = 255 #int-type raster no data is 255
                target_ds = None

                #save fixed temp arr to real geotiff
                ##update meta data to match the new wetland tif (i.e., 1 band instead of 3 in img.tif)
                wetland_meta = meta.copy()
                wetland_meta.update({'nbands': 1})


                wetland_tif = ra.array_to_geotif(temp_arr, wetland_meta, tile_dir,
                                                 tif_out.split('\\')[-1], dtype='int', nodata=255)
            feature = None
            source_ds = None
        else:
            print("No wetlands found in {}".format(tile_in))
            #close shp
            source_ds = None
            #move to a different directory
            #TODO: overwrite
            shutil.move(tile_dir, r"D:\3rdStudy_ONeil\data\s1_scratch\no_instances")



    print("Image directory created with {} images".format(len(img_subdirs)))


tif_in = r"D:\3rdStudy_ONeil\data\s1_scratch\comp.tif"

shp_in = r"D:\2ndStudy_ONeil\Tool_testing\data\Site1\wetlands.shp"

img_dir = r"D:\3rdStudy_ONeil\data\s1_scratch\imgs"



create_imgs(tif_in, shp_in, img_dir)