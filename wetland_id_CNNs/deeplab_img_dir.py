"""
Image directory home >> .\tensorflow\models\research\deeplab\datasets\wetlands
    +dataset
        +ImageSets --> contains train.txt, val.txt, trainval.txt
        +JPEGImages --> input color images (data), *.jpg
        +SegmentationClass --> ground truth annotations (wetland/BG binary) corresponding to each JPEGImage
    +tfrecord
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

def build_imgs(tif_in, shp_in, img_dir, tilesize=256):
    """
    :param tif_in: geotiff with n bands, each band representing input data
    :param shp_in: verification wetlands shapefile
    :param img_dir: ./datasets/wetlands/dataset
    :param tilesize: dimensions of images (pixels)
    :return: text file containing list of eligible images to use in train/test split
    """

    #define file path variables
    ImageSets = os.path.join(img_dir, "ImageSets")
    JPEGImages = os.path.join(img_dir, "JPEGImages")
    SegmentationClass = os.path.join(img_dir, "SegmentationClass")
    TifTiles = os.path.join(img_dir, "TifTemp")
    shp_dir = os.path.join(img_dir, "WetlandsSHP")

    # create text files if they do not exist
    eligImg = os.path.join(ImageSets, 'eligible.txt')

    #open geotiff and get info
    dset, meta = ra.gtiff_to_arr(tif_in, 'float')
    width = meta['ncol']
    height = meta['nrow']

    #first image #
    img_n = 0

    #cut composite.tif into tilesize x tilesize tiles
    for i in range(0, width, tilesize):
        for j in range(0, height, tilesize):
            w = min(i+tilesize, width) - i
            h = min(j+tilesize, height) - j
            #clip geotiff to extents and save as "[IMG#].tif" in imgdir
            cmd = "gdal_translate -of GTIFF -srcwin "+str(i)+", "+str(j)+", "+str(w)+", " \
                +str(h)+" " + tif_in + " " + TifTiles + r"\IMG{}.tif".format(img_n)

            os.system(cmd)

            #increase img #
            img_n +=1

    #clip shp to each tile extents
    for f_i in os.listdir(TifTiles):
        tile_in = os.path.join(TifTiles, f_i)

        tile_name = f_i[:-4]
        dset_tile, meta = ra.gtiff_to_arr(tile_in, 'float')

        #get tile info
        ulx, xres, xskew, uly, yskew, yres  = meta['ext']
        sizeX = meta['ncol'] * xres
        sizeY = meta['nrow'] * yres
        lrx = ulx + sizeX
        lry = uly + sizeY
        #TODO: fix errors in extent readings...works in code but may cause confusion - ymin and xmax are swithced

        # format the extent coords
        extent = '{0} {1} {2} {3}'.format(ulx, lry, lrx, uly)

        #shp named with tile number in corresponding directory
        shp_tile = os.path.join(shp_dir, tile_name+ ".shp")

        # make clip command with ogr2ogr - default to shapefile format
        cmd = 'ogr2ogr ' + shp_tile + ' ' + shp_in + ' -clipsrc ' + extent

        # call the command
        subprocess.call(cmd, shell=False)

        #open shapefile tile
        driver = ogr.GetDriverByName('ESRI Shapefile')
        source_ds = driver.Open(shp_tile, 1)
        source_layer = source_ds.GetLayer()

        #check if shapefile has wetlands
        feature_count = source_layer.GetFeatureCount()
        if feature_count > 0:

            #create JPEG image
            jpg = ra.gtiff_to_img(tile_in, JPEGImages, tile_name+".jpg", "JPG")

            # write filename to eligible.txt
            with open(eligImg, "a") as file:
                file.write("{}\n".format(jpg))
                file.close()

            #rasterize ground truth annotation
            gt_temp = "TEMP_"+ tile_name + ".tif"
            gt_out = tile_name + ".png"

            xmin, xmax, ymin, ymax = ulx, lrx, lry, uly  # see note above
            pixel_size = float(meta['pix_res'])

            # Create a dummy geotiff using the MEM driver of gdal
            target_ds = gdal.GetDriverByName('MEM').Create('', int(sizeX), int(sizeY * -1), gdal.GDT_Byte)
            target_ds.SetGeoTransform((xmin, pixel_size, 0, ymax, 0, -pixel_size))
            band = target_ds.GetRasterBand(1)
            band.SetNoDataValue(0)
            band.FlushCache()

            # Rasterize the wetland tile (not exported to a file)
            gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[1], options=['ALL_TOUCHED=TRUE'])
            # Read as temp wetland tile tif as array
            temp_arr = band.ReadAsArray()

            #force any no data areas from input data tile to be no data in gt annotated tiff
            temp_arr[dset_tile[:, :, 0] == -9999] = 255  # int-type raster no data is 255
            target_ds = None #close

            #save fixed temp arr to real geotiff
            ##update meta data to match the new wetland tif (i.e., 1 band instead of 3 in img.tif)
            wetland_meta = meta.copy()
            wetland_meta.update({'nbands': 1})
            gt_temp_tif = ra.arr_to_gtiff(temp_arr, wetland_meta, SegmentationClass,
                                             gt_temp, dtype='int', nodata=255)

            gt_png = ra.gtiff_to_img(gt_temp_tif, SegmentationClass, gt_out, "PNG")
            os.remove(gt_temp_tif) #delete the temp tif
            os.remove(gt_temp_tif+".aux.xml")

            source_ds = None #close wetland shp

        else:
            print("No wetlands found in {}".format(f_i))
            #close shp
            source_ds = None


    print("\nStudy site split into {} images\n".format(len(os.listdir(TifTiles))))
    print("{} images with wetland instances\n".format(len(os.listdir(SegmentationClass))))

    return eligImg

if __name__ == '__main__':

    tif_in = r"D:\2ndStudy_ONeil\Tool_testing\data\Site1\composites\comp_o_c.tif"
    shp_in = r"D:\2ndStudy_ONeil\Tool_testing\data\Site1\wetlands.shp"
    img_dir = r"D:\3rdStudy_ONeil\wetland_identification\wetland_id_CNNs\tensorflow\models\research\deeplab\datasets\wetlands\dataset"

    eligble_images = build_imgs(tif_in, shp_in, img_dir, tilesize=512)
