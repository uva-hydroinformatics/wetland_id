# -*- coding: utf-8 -*-
"""
Users of the wetland identification tool must use this script
to designate input files, tool directories, and parameters.

Created on Wed Mar 28 15:33:57 2018
@author: Gina O'Neil
"""

import os
import sys


#user must create data and results folders
tool_data = r"D:\2ndStudy_ONeil\Tool_testing\data" #all wetland tool data ***manually create***

tool_results = r"D:\2ndStudy_ONeil\Tool_testing\results" #all wetland tool results ***manually create***

#input files
input_dem = "lidar_1huc_clims.tif" #**key input**

#roi = study area specific to input dem, sub directories created for roi
roi = "Site4"

run_name = "Run1"

roi_data = os.path.join(tool_data, roi) #all data specific to roi

roi_results_dir = os.path.join(tool_results, roi)

roi_results = os.path.join(roi_results_dir, run_name) #results subdir: roi classification and accuracy results stored here

roi_wetlands = os.path.join(roi_data, "wetlands.shp")

roi_lim = os.path.join(roi_data, "lims.shp")

roi_proc_lim = os.path.join(roi_data, "WBD.shp") #optional, if dem not clipped to watershed boundary

roi_dems = os.path.join(roi_data, "DEM") #data subdir: DEM filtering outputs stored here

roi_vars = os.path.join(roi_data, "input_variables") #data subdir: processing outputs stored here

roi_geonet = os.path.join(roi_data, "geonet_out")

roi_comps = os.path.join(roi_data, "composites") #data subdir: composites of specific processing outputs stored here

roi_log = os.path.join(roi_results_dir, "log_files")

#names for output files
comp = "composite.tif" #change if trying multiple composites
rf_class = "rf_class.tif"
rf_fuzz = "rf_fuzzy.tif"
acc_report = "acc.xlsx"

pygeo_dir = r"D:\2ndStudy_ONeil\Scripts\wetland_identification"

dir_list = [roi_data, roi_dems, roi_vars, roi_comps, roi_results, roi_geonet, roi_log]

#create roi directories
for d in dir_list:
    if not os.path.exists(d):
        os.mkdir(d)
        
#***************************numerical parameters******************************#

slope_const = 0.0001 #value to be added to all slope values to avoid dividing by zero
n_proc = 8
beta = 4.0 #optional accuracy assessment output - Fbeta score i.e
pm_n_iter = 50 #number of Perona-Malik iterations
smoothing_width = 5. #scale of elevation smoothing (meters)
edge_thresh = 0.9 #Perona-Malik edge stopping threshold

#clipped tifs
clip_suf = "_c.tif"
#tilerinf params

gaus_stdev = float(smoothing_width / 4.)
wien_noise = None


dem_med = "med%sm.tif" %(str(smoothing_width).replace(".",""))
dem_mean = "mean%sm.tif" %(str(smoothing_width).replace(".",""))
dem_gaus = "gaus%s.tif" %(str(gaus_stdev).replace(".",""))
dem_wien = "wien%s.tif" %(str(smoothing_width))
dem_pm_slp = "pmSLP%s_%sp.tif" %(str(pm_n_iter), str(edge_thresh))
dem_pm_cv = "pmCV%s_%sp.tif" %(str(pm_n_iter), str(edge_thresh))

med_tif = os.path.join(roi_dems, dem_med)
mean_tif = os.path.join(roi_dems, dem_mean)
gaus_tif = os.path.join(roi_dems, dem_gaus)
pm_slp_tif = os.path.join(roi_dems, dem_pm_slp)
pm_cv_tif = os.path.join(roi_dems, dem_pm_cv)

pyg_channels = os.path.join(roi_geonet, input_dem[:-4]+"_channelNetwork.shp")


#outputs indpendent of conditioning 
slp_suf = "_slp.tif"
slp_nz_suf = "_slpnz.tif" #slope after adding small constant to all values
cv_suf = "_curv.tif"
dtw_suf = "_dtw.tif"

#conditioned dem filenames
z_fill_suf = "_fill.tif"
z_breach_suf = "_br.tif" #refers to IRA
z_at_suf = "_At.tif"

#***twi outputs preceded by dem conditioning***#
#fill calculations
slp_suf = "_slp.tif"
slpnz_suf = "_slp_add.tif"
fdr_suf = "_fdr.tif"
sca_suf = "_sca.tif"
twi_suf = "_twi.tif"

dtw_suf = "_grassrcost_dtw.tif"


#***********************taudem function paths*********************************#
taudem_fill = r"D:\2ndStudy_ONeil\Scripts\wetland_identification\td_fill.bat"
taudem_fdrslp = r"D:\2ndStudy_ONeil\Scripts\wetland_identification\td_dinf_fdr_slp.bat"
taudem_uca = r"D:\2ndStudy_ONeil\Scripts\wetland_identification\td_dinf_uca.bat"
taudem_TWI = r"D:\2ndStudy_ONeil\Scripts\wetland_identification\td_TWI.bat"
#TODO - find a way to pass these to the batch files used to run taudem...hardcoded in files now


#GRASS GIS parameters
#set grass7bin variable based on OS, enter correct path to grass72.bat file, or add path to file to PATH
def set_grassbin():
	if sys.platform.startswith('win'):
			# MS Windows
			grass7bin = r"C:\Program Files\GRASS GIS 7.2.2\grass72.bat"
			# uncomment when using standalone WinGRASS installer
			# grass7bin = r'C:\Program Files (x86)\GRASS GIS 7.2.0\grass72.bat'
			# this can be avoided if GRASS executable is added to PATH
	elif sys.platform == 'darwin':
			# Mac OS X
			# TODO: this have to be checked, maybe unix way is good enough
			grass7bin = '/Applications/GRASS/GRASS-7.2.app/'
	return grass7bin

location = "wetlandtool"
mapset = "wetlandtool_user"



"""
classification parameters:
    
    w_train_prop, nw_train_prop = proportion of w,nw verified data used to train model, complement /
                of this is automatically reserved to test model
    
    n_trees = number of estimators in RF model
    
    tree_depth = max. depth of each tree
    
    w_cw = class weight assigned to the positive (wetland) class
            positive pixel value = 0
    
    nw_cw = class weight of the negative (nonwetland) class
            negative pixel value = 1
    
"""
n_trees = 300
tree_depth = None 
wcw = 'balanced'
nwcw = 'balanced'
w_train_prop = 0.15 
nw_train_prop = 0.10
