# -*- coding: utf-8 -*-
"""
Created on Thu Feb 08 19:51:05 2018

@author: Linnea
"""

import sklearn_class_and_acc as sk
#import model_eval_experimentation as me
import os
#import me_1 as me1

results_dir = r"C:\Users\Linnea\Downloads\sklearn_files" 

# composite is a multidimensional raster file where each dimensions represents an input variable (i.e., input variable rasters are "stacked" into one)
composite = "RF_A_clipenvebm.tif"
#train_tif is the training raster file (and path to) used in the RF classification
train_tif = "50p_100ar3_snap.tif"
#known delineations:
true_ds = "p06_vals_str_clip.tif"


composite_input = os.path.join(results_dir,composite)
train_input = os.path.join(results_dir,train_tif)
true_input = os.path.join(results_dir,true_ds)

sk.main(train_input,composite_input, true_input)
