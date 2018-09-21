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


tif_in = r"D:\3rdStudy_ONeil\data\s1_scratch\comp.tif"
imgs_dataset = r"D:\3rdStudy_ONeil\data\s1_scratch\imgs"
dset = gdal.Open(tif_in)
ext = dset.GetGeoTransform()

width = dset.RasterXSize
height = dset.RasterYSize

print (width, 'x', height)

tilesize = 500

img_n = 0

for i in range(0, width, tilesize):
    for j in range(0, height, tilesize):
        new_dir = r"D:\3rdStudy_ONeil\data\s1_scratch\imgs\%s" %(str(img_n))
        os.mkdir(new_dir)
        w = min(i+tilesize, width) - i
        h = min(j+tilesize, height) - j
        gdaltranString = "gdal_translate -of GTIFF -srcwin "+str(i)+", "+str(j)+", "+str(w)+", " \
            +str(h)+" " + tif_in + " " + new_dir + r"\%d.tif" %(img_n)
        os.system(gdaltranString)
        img_n +=1

inVectorPath = r"D:\2ndStudy_ONeil\Tool_testing\data\Site1\wetlands.shp"
img_dirs = next(os.walk(imgs_dataset))[1]

for subdir in img_dirs:
    inRasterPath = os.path.join(imgs_dataset, subdir)
    files = [f for f in os.listdir(inRasterPath) if f.endswith(".tif")]
    for f in files:
        src = gdal.Open(os.path.join(inRasterPath, f))
        ulx, xres, xskew, uly, yskew, yres  = src.GetGeoTransform()
        sizeX = src.RasterXSize * xres
        sizeY = src.RasterYSize * yres
        lrx = ulx + sizeX
        lry = uly + sizeY
        src = None
        outVectorPath = os.path.join(inRasterPath, "%s_wetland.shp" %(f[:-4]))
        # format the extent coords
        extent = '{0} {1} {2} {3}'.format(ulx, lry, lrx, uly);

        # make clip command with ogr2ogr - default to shapefile format
        cmd = 'ogr2ogr ' + outVectorPath + ' ' + inVectorPath + ' -clipsrc ' + extent

        # call the command
        subprocess.call(cmd, shell=True)

        driver = ogr.GetDriverByName('ESRI Shapefile')
        fn = outVectorPath
        dataSource = driver.Open(fn, 1)

        layer = dataSource.GetLayer()
        feature_count = layer.GetFeatureCount()

        if feature_count > 0:
            defn = layer.GetLayerDefn()
            field_count=defn.GetFieldCount()
            id_field = ogr.FieldDefn("instance", ogr.OFTInteger)
            layer.CreateField(id_field)
            feature = layer.GetNextFeature()
            x = 0
            while feature is not None:
                feature.SetField("instance", x)
                layer.SetFeature(feature)
                tif_out = outVectorPath[:-4] + "_%d.tif" % (x)
                cmd = 'gdal_rasterize -burn 1 -where \"instance=%d\" -a_nodata 0 -ot Int32 -tr 1 1 \"%s\" \"%s\"' \
                    %(x, outVectorPath, tif_out)
                print (cmd)
                subprocess.Popen(cmd)
                feature.Destroy()
                feature = layer.GetNextFeature()
                x += 1
            feature = None

        else:
            driver.DeleteDataSource(outVectorPath)

        dataSource.Destroy()
