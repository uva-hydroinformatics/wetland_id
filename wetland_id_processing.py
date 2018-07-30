# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 15:33:57 2018

This program generates wetland predictions from high-resolution DEM data using the
Random Forest classification method. We use modules from PyGeoNet, 
the python implementation of GeoNet 
(see Sangireddy et al. for details http://dx.doi.org/10.1016/j.envsoft.2016.04.026),
and GDAL, SKlearn, and Scipy to execute the automated workflow. Main program executes
all unique combinations of tested filtering and conditioning techniques. Future development
of this model will implement only the best performing filtering and conditioning. 


author: Gina O'Neil
"""
import wetland_id_defaults as default
from raster_array_funcs import *
import class_acc_funcs 
import filtering
import create_inputs
import os
import warnings 
import numpy as np
import sys
import warnings
import pygeonet_processing_v2
import datetime as dt
from scipy import ndimage

def main():
    
    warnings.filterwarnings("ignore",category=RuntimeWarning)
    
    #create log file for run
    now = dt.datetime.now()
    dt_str = now.strftime("%y%m%d_%H%M")
    f = open(os.path.join(default.roi_log, 'log_%s_%s.txt'%(default.roi, dt_str)), 'w')
    sys.stdout = f
    print "Program starting! Start time: %s \n" %(dt_str) 
    
    #******************************open and clean input dem tif**********************************
    
    dem_in = os.path.join(default.roi_dems, default.input_dem)
    dem_arr, dem_meta = geotif_to_array(dem_in)

    pix_res = float(dem_meta['pix_res'])
    dem_arr_mask = clean_array(dem_arr)
    dem_arr_nans = np.ma.filled(dem_arr_mask, np.nan)
    
    """We execute PyGeoNet to extract channel centerlines for DTW calculation. PyGeoNet code was 
    downloaded from  https://sites.google.com/site/geonethome/source-code(Sangireddy et al., 2016) 
    in January, 2018 and slightly modified without changing the nature of calculation."""
    ##run pygeonet to extract channels from dem
    pyg_channels = pygeonet_processing_v2.main(dem_arr_nans)
    
    ##**********perform filtering - should pass the masked, clean dem to filter modules***********
    
    #as intended
    dem_mean, dem_med, dem_gaus, dem_pmslp = filtering.main(dem_arr_mask, dem_meta)
    
    ##NOTE: if ROI are near the edge of HUC12 extents, it will be necessary to remove edges effected by filtering methods \
    #best if ROI shp is edited to include non-erroneous data
    clip_bounds = default.roi_lim
    
    ####*******************for each filtered dem, create inputs********************************************
    
    dem_list = [dem_arr_nans, dem_mean, dem_med, dem_gaus, dem_pmslp] #these are the geotiffs
    
    for dem in dem_list:        

        dem_fname = dem.split('\\')[-1]
            
        rootname = dem_fname[:-4]
    
        dem_arr, dem_meta = geotif_to_array(dem)
    
        dem_arr_mask = clean_array(dem_arr)    
    
        #**********calculate slope and curvature for each filtered dem, pass masked dem*************
        slp_arr, curv_arr = create_inputs.calc_slope_curv(dem_arr_mask, pix_res)
        
        slp_tif = array_to_geotif(slp_arr, dem_meta, default.roi_vars, rootname + default.slp_suf, nodata=0)
            
        cv_tif = array_to_geotif(curv_arr, dem_meta, default.roi_vars, rootname + default.cv_suf)
    
        del slp_arr, curv_arr
    
        dtw, dtw_dir = create_inputs.calc_dtw(slp_tif, pyg_channels)
    
        
        #*********perform hydro conditioning and calc TWI for each filtered dem***********    
        
        dem_fill = create_inputs.calc_fill(dem, default.roi_dems)
        fill_slp, fill_fdr = create_inputs.calc_fdr(dem_fill, default.roi_vars)
        fill_sca = create_inputs.calc_sca(fill_fdr)
        fill_twi = create_inputs.calc_twi(slp_tif, fill_sca)
       
        dem_ira = create_inputs.calc_ira(dem)
        ira_slp, ira_fdr = create_inputs.calc_fdr(dem_ira, default.roi_vars)
        ira_sca = create_inputs.calc_sca(ira_fdr)
        ira_twi = create_inputs.calc_twi(slp_tif, ira_sca)
#        ira_twi = create_inputs.calc_ira_twi(dem) ##this is an alternative, calculating with only GRASS but takes MUCH longer
    
        #A* algorithm makes no changes to dem...instead calc and return TWI and its inputs all at once
        at_twi = create_inputs.calc_at_twi(dem)
        
    ##***********************create composite files************************************##
    ##all of these are only needed for preprocessing technique experiements

    wdir = default.roi_vars
    outdir = default.roi_comps
    root_dem = default.input_dem[:-4]
    
    #no filter + fill
    twi = os.path.join(wdir, root_dem + "_fill_twi.tif")
    curv = os.path.join(wdir, root_dem + "_curv.tif")
    dtw = os.path.join(wdir, root_dem + "_slp_grassrcost_dtw.tif")
    flist = [twi, curv, dtw]
    comp = create_inputs.comp_bands(outdir, flist, "comp_a.tif")
    
    #no filter + ira
    twi = os.path.join(wdir, root_dem + "_grass_breach_twi.tif")
    curv = os.path.join(wdir, root_dem + "_curv.tif")
    dtw = os.path.join(wdir, root_dem + "_slp_grassrcost_dtw.tif")
    flist = [twi, curv, dtw]
    comp = create_inputs.comp_bands(outdir, flist, "comp_b.tif")                                       
    
    #no filter + A*
    twi = os.path.join(wdir, root_dem + "_grass_at_twi.tif")
    curv = os.path.join(wdir, root_dem + "_curv.tif")
    dtw = os.path.join(wdir, root_dem + "_slp_grassrcost_dtw.tif")
    flist = [twi, curv, dtw]
    comp = create_inputs.comp_bands(outdir, flist, "compc.tif")
    
    #mean filter + fill
    twi = os.path.join(wdir, "mean5m_fill_twi.tif")
    curv = os.path.join(wdir, "mean5m_curv.tif")
    dtw = os.path.join(wdir, "mean5m_slp_grassrcost_dtw.tif")
    flist = [twi, curv, dtw]
    comp = create_inputs.comp_bands(outdir, flist, "comp_d.tif")
    
    #mean filter + ira
    twi = os.path.join(wdir, "mean5m_grass_breach_twi.tif")
    curv = os.path.join(wdir, "mean5m_curv.tif")
    dtw = os.path.join(wdir, "mean5m_slp_grassrcost_dtw.tif")
    flist = [twi, curv, dtw]
    comp = create_inputs.comp_bands(outdir, flist, "comp_e.tif")

    #mean filter + A*    
    twi = os.path.join(wdir, "mean5m_grass_at_twi.tif")
    curv = os.path.join(wdir, "mean5m_curv.tif")
    dtw = os.path.join(wdir, "mean5m_slp_grassrcost_dtw.tif")
    flist = [twi, curv, dtw]
    comp = create_inputs.comp_bands(outdir, flist, "comp_f.tif")
    
    #median filter + fill
    twi = os.path.join(wdir, "med5m_fill_twi.tif")
    curv = os.path.join(wdir, "med5m_curv.tif")
    dtw = os.path.join(wdir, "med5m_slp_grassrcost_dtw.tif")
    flist = [twi, curv, dtw]
    comp = create_inputs.comp_bands(outdir, flist, "comp_g.tif")
    
    #median filter + ira
    twi = os.path.join(wdir, "med5m_grass_breach_twi.tif")
    curv = os.path.join(wdir, "med5m_curv.tif")
    dtw = os.path.join(wdir, "med5m_slp_grassrcost_dtw.tif")
    flist = [twi, curv, dtw]
    comp = create_inputs.comp_bands(outdir, flist, "comp_h.tif")
    
    #median filter + A*
    twi = os.path.join(wdir, "med5m_grass_at_twi.tif")
    curv = os.path.join(wdir, "med5m_curv.tif")
    dtw = os.path.join(wdir, "med5m_slp_grassrcost_dtw.tif")
    flist = [twi, curv, dtw]
    comp = create_inputs.comp_bands(outdir, flist, "comp_i.tif")
    
    #gaus filter + fill
    twi = os.path.join(wdir, "gaus125_fill_twi.tif")
    curv = os.path.join(wdir, "gaus125_curv.tif")
    dtw = os.path.join(wdir, "gaus125_slp_grassrcost_dtw.tif")
    flist = [twi, curv, dtw]    
    comp = create_inputs.comp_bands(outdir, flist, "comp_j.tif")

    #gaus filter + ira    
    twi = os.path.join(wdir, "gaus125_grass_breach_twi.tif")
    curv = os.path.join(wdir, "gaus125_curv.tif")
    dtw = os.path.join(wdir, "gaus125_slp_grassrcost_dtw.tif")
    flist = [twi, curv, dtw]
    comp = create_inputs.comp_bands(outdir, flist, "comp_k.tif")
    
    #gaus filter + A*
    twi = os.path.join(wdir, "gaus125_grass_at_twi.tif")
    curv = os.path.join(wdir, "gaus125_curv.tif")
    dtw = os.path.join(wdir, "gaus125_slp_grassrcost_dtw.tif")
    flist = [twi, curv, dtw]
    comp = create_inputs.comp_bands(outdir, flist, "comp_l.tif")
    
    #pm filter + fill
    twi = os.path.join(wdir, "pmSLP50_0.9p_fill_twi.tif")
    curv = os.path.join(wdir, "pmSLP50_0.9p_curv.tif")
    dtw = os.path.join(wdir, "pmSLP50_0.9p_slp_grassrcost_dtw.tif")
    flist = [twi, curv, dtw]
    comp = create_inputs.comp_bands(outdir, flist, "comp_m.tif")
    
    #pm filter + ira
    twi = os.path.join(wdir, "pmSLP50_0.9p_grass_breach_twi.tif")
    curv = os.path.join(wdir, "pmSLP50_0.9p_curv.tif")
    dtw = os.path.join(wdir, "pmSLP50_0.9p_slp_grassrcost_dtw.tif")
    flist = [twi, curv, dtw]
    comp = create_inputs.comp_bands(outdir, flist, "comp_n.tif")
    
    #pm filter + A*
    twi = os.path.join(wdir, "pmSLP50_0.9p_grass_at_twi.tif")
    curv = os.path.join(wdir, "pmSLP50_0.9p_curv.tif")
    dtw = os.path.join(wdir, "pmSLP50_0.9p_slp_grassrcost_dtw.tif")
    flist = [twi, curv, dtw]
    comp = create_inputs.comp_bands(outdir, flist, "comp_o.tif")
    
    
    comps_noclip = [os.path.join(default.roi_comps, c) for c in os.listdir(default.roi_comps)]
    comps = []
    
    for c in comps_noclip:
        clip = clip_geotif(c, default.roi_comps, default.roi_lim)
        comps.append(clip)
        
    #####**************************classification and accuracy assessment********************##

    ####use to create verif data from wetlands shp...CHECK THAT ALL PROJECTIONS ARE THE SAME
    var_arr, var_meta = geotif_to_array(os.path.join(default.roi_comps, "comp_o.tif")) #can be any composite file
    verif_temp = create_verif(default.roi_wetlands, default.roi_lim, default.roi_data, pix_res)
    verif_tif = clip_geotif(verif_temp, default.roi_data, default.roi_lim)
        
    results_dir = default.roi_results
    
    verif_arr, verif_meta = geotif_to_array(verif_tif)
    nans = np.where(np.isnan(verif_arr))

    #preprocessing combos used same training and testing data
    train_labels_2d, test_labels_2d = create_tt_labels(verif_tif, default.w_train_prop, n, results_dir)
        
    for comp in comps:
        rf_out, fuz_out, results = class_acc_funcs.main(comp, verif_tif)
    
        var_arr, var_meta = geotif_to_array(comp)
    
        train_features, test_features = create_tt_feats(comp, train_labels_2d, test_labels_2d, results_dir)
    
        rf_predict, fuzzy_predict_w, importance = classify(train_features, train_labels_2d, test_features, default.n_trees, default.tree_depth, default.wcw, default.nwcw)
    
        rf_predict[nans] = np.nan
        fuzzy_predict_w[nans] = np.nan
    
        comp_base = comp.split('\\')[-1]
    
        rf_results_name = "%s_rf_predict_w%.3f_nw%.3f_cws_%s_%s.tif" \
        %(comp_base[:-4], 100*default.w_train_prop, 100*n, str(default.wcw), str(default.nwcw))
        
        fuz_results_name = "%s_fuzzy_predict_w%.3f_nw%.3f_cws_%s_%s.tif" \
        %(comp_base[:-4], 100*default.w_train_prop, 100*n, str(default.wcw), str(default.nwcw))
    
        rf_out = array_to_geotif(rf_predict, verif_meta,results_dir, rf_results_name)
        fuz_out = array_to_geotif(fuzzy_predict_w, verif_meta, results_dir, fuz_results_name)
    
        results_fname = os.path.join(results_dir, "%s_w%.3f_nw%.3f_cws_%s_%s.xlsx" \
                                     %(comp_base[:-4], 100*default.w_train_prop, 100*n, str(default.wcw), str(default.nwcw)))   
        
        results = get_acc(test_labels_2d, rf_predict, fuzzy_predict_w, importance, results_fname, results_dir, var_meta)

    
    now = dt.datetime.now()
    dt_str = now.strftime("%y%m%d_%H%M")
    print "Program Complete! End time: %s" %(dt_str)
    sys.exit('Program complete!')
    sys.stdout.close()

if __name__ == '__main__':
    main()
    sys.exit('exit')
