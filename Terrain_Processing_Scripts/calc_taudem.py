# -*- coding: utf-8 -*-
"""
This program utilizes the TauDEM library to calculate TWI and its components, i.e.,
-Slope
-flow directions
-upstream contributing area
-Topographic wetness index.
These outputs are calcualted using the D8 flow direction method (used in ArcGIS)
as well as with the D-infinity flow direction method (developed by D. Tarbarton).
Batch files are called to execute these calculations, calculating on 8 processors
(can be changed in batch files).

Before processing,the user designated elevation file is projected to UTM (currently VA 17N). 

Download TauDEM: http://hydrology.usu.edu/taudem/taudem5/
Download gdal: https://github.com/creare-com/pydem/blob/master/INSTALL.md

Note 1: if downloading gdal with Conda, must manually add path environment varaible
Note 2: Batch files calling Dinf-specific commands must include file path to the
        Dinf executable, this was not necessary for D8-specific or general TauDEM
        executables.

Command-line args: input elevation file, Datum, UTM zone #, UTM zone N/S

Author: Gina O'Neil
"""

from osgeo import gdal, osr
import os
import subprocess
import sys

f_in = sys.argv[1] #input elevation file

def Check_prj(filename):
    gdal_dset = gdal.Open(filename)
    prj = gdal_dset.GetProjection()
    srs = osr.SpatialReference(wkt = prj)
    if srs.IsProjected:
        pcs = srs.GetAttrValue('projcs')
        print pcs
        if pcs == "%s / UTM zone %s%s" %(sys.argv[2], sys.argv[3], sys.argv[4]):
            print "Projection is: {} \n".format(prj)
            prj_file = filename
            return prj_file
        else:
            prj_file = filename[:-4] + '_prj.tif'
            print "Projected coordinate system does not match user input\n"
            print ("Projecting %s to %s UTM ZONE %s%s..................\n" \
                   %(filename, sys.argv[2], sys.argv[3], sys.argv[4]))
            cmd = 'gdalwarp.exe %s %s -t_srs "+proj=utm +zone=%s +datum=%s" -tr 0.76200152 0.76200152' \
                %(filename, prj_file, sys.argv[3], sys.argv[2])
            cmd_info = 'gdalinfo.exe -stats %s'%(prj_file)
            subprocess.call(cmd, shell = True)
            subprocess.call(cmd_info, shell = True)
            return prj_file
    else:
        prj_file = filename[:-4] + '_prj.tif'
        print "Projection is: {} \n".format(prj)   
        print ("Projecting %s to %s UTM ZONE %s%s..................\n" \
               %(filename, sys.argv[2], sys.argv[3], sys.argv[4]))
        cmd = 'gdalwarp.exe %s %s -t_srs "+proj=utm +zone=%s +datum=%s" -tr 0.76200152 0.76200152' \
            %(filename, prj_file, sys.argv[3], sys.argv[2])
        cmd_info = 'gdalinfo.exe -stats %s'%(prj_file)
        subprocess.call(cmd, shell = True)
        subprocess.call(cmd_info, shell = True)
        return prj_file

def remove_pits(prj_file):
    fel_file = prj_file[:-4]+"fel.tif" #this is the default file name given to the TauDEM remvove pits output
    if os.path.exists(fel_file):
        print "Elevation file with pits removed already exists! \n"
        print "Filled elevation file: %s \n" %(fel_file)
        return fel_file
    else:
        print "Filled elevation file does not exist! \n"
        print "Removing pits................................................\n"
        subprocess.call("remove_pits_n8.bat %s" %(prj_file)) #args = [elev.tif]
        return fel_file

def D8_calcs(fel_file):        
    D8_fdr = '.\D8\\' + fel_file[:-4]+"D8_fdr.tif"
    D8_slp = '.\D8\\' + fel_file[:-4]+"D8_slp.tif"
    D8_uca = '.\D8\\' + fel_file[:-4]+"D8_uca.tif"
    D8_TWI = '.\D8\\' + fel_file[:-4]+"D8_twi.tif"
    
    print "Executing D8 calculations........................................\n"
    
    if not os.path.exists('D8'):
        os.mkdir('D8')
    
    if not os.path.exists(D8_fdr):
        print "Calculating D8 flow directions and slope.....................\n"
        subprocess.call("D8FlowDir_n8.bat %s %s %s" %(D8_fdr, D8_slp, fel_file))
    else:
        print "D8 flow directions and slope already exist! \n"
        
    if not os.path.exists(D8_uca):
        print "Calculating D8 upstream constributing area...................\n"
        subprocess.call("D:\PyDEM_testing\USGS_tiff\TauDEM\D8_UCA_n8.bat %s %s" %(D8_fdr, D8_uca))
    else:
        print "D8 upstream contributing area already exists! \n"
    
    if not os.path.exists(D8_TWI):
        print "Calculating TWI from D8 components...........................\n"
        subprocess.call("D:\PyDEM_testing\USGS_tiff\TauDEM\TWI_n8.bat %s %s %s" %(D8_slp, D8_uca, D8_TWI))
    else:
        print "TWI from D8 components already exists! \n"
    
    print "D8 calculations complete! \n"

def Dinf_calcs(fel_file):        
    Dinf_fdr = '.\Dinf\\' + fel_file[:-4]+"Dinf_fdr.tif"
    Dinf_slp = '.\Dinf\\' + fel_file[:-4]+"Dinf_slp.tif"
    Dinf_uca = '.\Dinf\\' + fel_file[:-4]+"Dinf_uca.tif"
    Dinf_TWI = '.\Dinf\\' + fel_file[:-4]+"Dinf_twi.tif"
    
    print "Executing Dinf calculations......................................\n"
    
    if not os.path.exists('Dinf'):
        os.mkdir('Dinf')
    
    if not os.path.exists(Dinf_fdr):
        print "Calculating Dinf flow directions and slope...................\n"
        subprocess.call("DinfFlowDir_n8.bat %s %s %s" %(Dinf_fdr, Dinf_slp, fel_file))
    else:
        print "Dinf flow directions and slope already exist! \n"
        
    if not os.path.exists(Dinf_uca):
        print "Calculating Dinf upstream constributing area.................\n"
        subprocess.call("D:\PyDEM_testing\USGS_tiff\TauDEM\Dinf_UCA_n8.bat %s %s" %(Dinf_fdr, Dinf_uca))
    else:
        print "D8 upstream contributing area already exists! \n"
    
    if not os.path.exists(Dinf_TWI):
        print "Calculating TWI from Dinf components.........................\n"
        subprocess.call("D:\PyDEM_testing\USGS_tiff\TauDEM\TWI_n8.bat %s %s %s" %(Dinf_slp, Dinf_uca, Dinf_TWI))
    else:
        print "TWI from Dinf components already exists! \n"
        
    print "Dinf calculations complete! \n"

def clip_rasters(raster, boundary): #must pass filename w/ path of raster, not an opened gdal dset
    # Name of clip raster file(s)
    clipped = raster[:-4]+'_clip.tif'
    
    cmd = "gdalwarp.exe -cutline %s -crop_to_cutline -dstnodata -9999.0 \
    %s %s" %(boundary, raster, clipped) 
    
    cmd_info = 'gdalinfo.exe -stats %s'%(clipped)
    subprocess.call(cmd)
    subprocess.call(cmd_info)
    
print "%s has been clipped!" %(raster)

        
def main():
     
    #input elevation file
    f_in = sys.argv[1]
    
    #check projection of elevation file
    filename_to_elevation_geotiff = Check_prj(f_in)

    #remove pits from projcted elevation file
    fel_file = remove_pits(filename_to_elevation_geotiff)

    #execute D8 calcs
    D8_calcs(fel_file)
    
    #execute Dinf calcs
    Dinf_calcs(fel_file)

    #clip outputs to boundary    
    shp = r"G:\P06_RT29\Data\WBD\WBD_prj.shp"
    
    for tiff in os.listdir('.\D8'):
        clip_rasters('.\D8\\'+tiff, shp)
    
    for tiff in os.listdir('.\Dinf'):
        clip_rasters('.\Dinf\\'+tiff, shp)

    print "Done!"


if __name__ == '__main__':
    main()
