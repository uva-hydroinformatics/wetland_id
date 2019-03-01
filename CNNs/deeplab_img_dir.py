"""
Image directory home >> .\tensorflow\models\research\deeplab\datasets\wetlands
    +dataset
        +ImageSets --> contains train.txt, val.txt, trainval.txt
        +JPEGImages --> input color images (data), *.jpg
        +SegmentationClass --> ground truth annotations (wetland/BG binary) corresponding to each JPEGImage
    +tfrecord

Author: Gina O'Neil
Initial Version: Feb. 13, 2019
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
import math
from random import shuffle
import sklearn.preprocessing as skp

def build_imgs(tif_in, wetlands_shp, bounds_shp, img_dir, tilesize, no_data_val):
    """
    :param tif_in: geotiff with n bands, each band representing input data
    :param wetlands_shp: verification wetlands shapefile
    :param bounds_shp: limits of verification wetlands
    :param img_dir: ./datasets/wetlands/dataset
    :param tilesize: dimensions of images (pixels)
    :param no_data_val: no data value to assign to output images, will be same for input data images and annotations
    :return: text file containing list of eligible images to use in train/test split
    """

    nonw_val = 0
    w_val = 1

    #define file path variables
    ImageSets = os.path.join(img_dir, "ImageSets")
    JPEGImages = os.path.join(img_dir, "JPEGImages")
    SegmentationClass = os.path.join(img_dir, "SegmentationClass")
    IMG_Tiles = os.path.join(img_dir, "ImgsTemp")
    GT_Tiles = os.path.join(img_dir, "WetlandsTemp")

    # create text files if they do not exist
    eligImg = os.path.join(ImageSets, 'trainval.txt')
    if os.path.exists(eligImg):
        os.remove(eligImg)

    
    #open orig geotiff and get info
    dset, meta = ra.gtiff_to_arr(tif_in, 'float')

    #rasterize wetland and nonwetland area    
    inDriver = ogr.GetDriverByName("ESRI Shapefile")
    inDataSource = inDriver.Open(bounds_shp, 0)
    inLayer = inDataSource.GetLayer()
    bounds_ext = inLayer.GetExtent()
    
    gt_tif_temp = os.path.join(img_dir, "annotationTEMP.tif")

    list_options = [
        '-init {}'.format(nonw_val),
        '-burn {}'.format(w_val),
        '-a_nodata 255',#.format(no_data_val),
        '-ot Byte',
        '-co COMPRESS=LZW',
        '-te {} {} {} {}'.format(bounds_ext[0], bounds_ext[2], bounds_ext[1], bounds_ext[3]),
        '-tr {} {}'.format(meta['pix_res'], meta['pix_res'])
    ]
    options_string = " ".join(list_options)
    gdal.Rasterize(gt_tif_temp, wetlands_shp, options=options_string)
    
    #get correct no data vals
    gt_tif = ra.clip_geotif(gt_tif_temp, img_dir, bounds_shp, no_data=no_data_val)
    os.remove(gt_tif_temp)

    width = meta['ncol']
    height = meta['nrow']

    #first image #
    img_n = 0

    #cut composite.tif into tilesize x tilesize tiles
    for i in range(0, width, tilesize):
        for j in range(0, height, tilesize):
            w = min(i+tilesize, width) - i
            h = min(j+tilesize, height) - j

            #clip data geotiff to extents and save as "[IMG#].tif" in imgdir
            cmd = "gdal_translate -of GTIFF -srcwin "+str(i)+", "+str(j)+", "+str(w)+", " \
                +str(h)+" " + tif_in + " " + IMG_Tiles + r"\IMG{}.tif".format(img_n)

#            os.system(cmd)
            subprocess.call(cmd, shell=True)

            #clip annotation geotiff to extents and save as "[IMG#].tif" in imgdir            
            cmd = "gdal_translate -of GTIFF -srcwin "+str(i)+", "+str(j)+", "+str(w)+", " \
                +str(h)+" " + gt_tif + " " + GT_Tiles + r"\IMG{}.tif".format(img_n)


#            os.system(cmd)
            subprocess.call(cmd, shell=True)
            
            #increase img #
            img_n +=1

    #clip shp to each tile extents
    for f_i in os.listdir(IMG_Tiles):

        img_in = os.path.join(IMG_Tiles, f_i)
        img_name = f_i[:-4]

        dset_tile, meta = ra.gtiff_to_arr(img_in, 'float')
        
        gt_in = os.path.join(GT_Tiles, f_i)
        gt_name = f_i[:-4]

        gt_tile, _ = ra.gtiff_to_arr(gt_in, 'byte')

        #get tile info
        ulx, xres, xskew, uly, yskew, yres  = meta['ext']
        sizeX = meta['ncol'] * xres
        sizeY = meta['nrow'] * yres
        lrx = ulx + sizeX
        lry = uly + sizeY
        #TODO: fix errors in extent readings...works in code but may cause confusion - ymin and xmax are swithced

        # format the extent coords
        extent = '{0} {1} {2} {3}'.format(ulx, lry, lrx, uly)

        #check if wetlands exist in GT data and do not convert image tiles with no data (-9999)
        if w_val in np.unique(gt_tile) and np.min(dset_tile[:, :, 0]) != -9999.:

            #create JPEG image
            jpg = ra.gtiff_to_img(img_in, JPEGImages, img_name+".jpg", "JPG", no_data_val)

            #create PNG image
            gt_png = ra.gtiff_to_img(gt_in, SegmentationClass, img_name+".png", "PNG", no_data_val)
            
            # write filename to eligible.txt
            with open(eligImg, "a") as file:
                file.write("{}\n".format(os.path.basename(jpg)[:-4])) #to be compatible with existing deeplab scripts, only save basename w/o extension
                file.close()

#            #rasterize ground truth annotation
#            gt_temp = "TEMP_"+ img_name + ".tif"
#            gt_out = img_name + ".png"
#
#            xmin, xmax, ymin, ymax = ulx, lrx, lry, uly  # see note above
#            pixel_size = float(meta['pix_res'])
#
#            # Create a dummy geotiff using the MEM driver of gdal
#            target_ds = gdal.GetDriverByName('MEM').Create('', int(sizeX), int(sizeY * -1), gdal.GDT_UInt16)#gdal.GDT_Byte)
#            target_ds.SetGeoTransform((xmin, pixel_size, 0, ymax, 0, -pixel_size))
#            band = target_ds.GetRasterBand(1)
#            band.Fill(nonwetland_val) # pixels outside of polygon will be filled with this val
#            band.FlushCache()
#
#            # Rasterize the wetland tile (not exported to a file)
#            gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[wetland_val], options=['ALL_TOUCHED=TRUE'])
#            # Read as temp wetland tile tif as array
#            temp_arr = band.ReadAsArray()
#
#            #force any no data areas from input data tile to be no data in gt annotated tiff
#            temp_arr[dset_tile[:, :, 0] == -9999] = no_data_val  # int-type raster no data is 0
#            target_ds = None #close
#
#            #save fixed temp arr to real geotiff
#            ##update meta data to match the new wetland tif (i.e., 1 band instead of 3 in img.tif)
#            wetland_meta = meta.copy()
#            wetland_meta.update({'nbands': 1})
#            gt_temp_tif = ra.arr_to_gtiff(temp_arr, wetland_meta, SegmentationClass,
#                                             gt_temp, dtype='int', nodata=no_data_val)
#
#            gt_png = ra.gtiff_to_img(gt_temp_tif, SegmentationClass, gt_out, "PNG", no_data_val)
#            os.remove(gt_temp_tif) #delete the temp tif
#            os.remove(gt_temp_tif+".aux.xml")
#
#            source_ds = None #close wetland shp

        else:
            print("No wetlands found in {}".format(f_i))
            #close shp
#            source_ds = None


    print("\nStudy site split into {} images\n".format(len(os.listdir(IMG_Tiles))))
    print("{} images with wetland instances\n".format(len([f for f in os.listdir(SegmentationClass) if f.endswith(".png")])))

    return eligImg

def split_imgs(ImgList, trainImg, valImg, train_percent):
    """
    :param ImgList: text file containing list of all eligible image files to use
    :param trainImg: filepath to text file to save training images
    :param valImg: filepath to text file to save validation images
    :param train_percent: percent of ImgList to pick for training
    :return: trainImg and valImg lists
    """
    val_percent = 1 - (train_percent)

    # from list of eligible images, randomly split into train, val, and trainval
    with open(ImgList, "r") as text:
        elig_list = text.read().splitlines()
    text.close()

    shuffle(elig_list)

    # don't allow overwriting or appending existing text files
    check_list = [trainImg, valImg]
    for f in check_list:
        if os.path.exists(f):
            os.remove(f)

    trainImgThresh = math.ceil(len(elig_list) * train_percent)
    trainList = elig_list[: trainImgThresh]
    with open(trainImg, "w") as text:
        text.writelines('\n'.join(t for t in trainList))
    text.close()

    valImgThresh = int(len(elig_list) * val_percent)
    valList = elig_list[ -valImgThresh : ]
    with open(valImg, "w") as text:
        text.writelines('\n'.join(t for t in valList))
    text.close()

    return trainList, valList

