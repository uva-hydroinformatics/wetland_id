"""
This script executes the DeepLab V3 semantic segmentation workflow for wetland detection from LiDAR topography-derived
input variables. From an input variable geotiff, a user-defined number of image tiles are created. These image tiles are
then randomly split into training, training validation, and validation image sets. All images are converted to
tfrecords, which are then used to train the DeepLab model. DeepLab tensorflow code was cloned from:
https://github.com/tensorflow/models/tree/master/research/deeplab, and the tutorial from:
https://medium.freecodecamp.org/how-to-use-deeplab-in-tensorflow-for-object-segmentation-using-deep-learning-a5777290ab6b
was used to guide changes to the DeepLab library needed to make compatible with the wetland dataset.

Author: Gina O'Neil
Initial Version: Feb. 14, 2019
"""
from deeplab_img_dir import build_imgs
from deeplab_img_dir import split_imgs
import os
import subprocess
import sys

#=======================================================================================================================
#Before running, manually change:
#
# - number of train, trainval, and val images in segmentation_dataset.py (L113-L117)
# - model hyperparams in FLAGS of train.wetlands.sh
# - NUM_ITERATIONS in train_wetlands.sh (L18)
# - edit --tf_initial_checkpoint FLAG in train_wetlands.sh AND train.py if model is training from a checkpoint
# - remove/edit "source activate [VENV]" in sh scripts
#
# **note that ./tensorflow/models/research/deeplab/utils/train_utils.py was modified to apply imbalanced weights
# to classes
#=======================================================================================================================

#filepaths
tif_in = r"D:\2ndStudy_ONeil\Tool_testing\data\Site1\composites\comp_o_c.tif"
shp_in = r"D:\2ndStudy_ONeil\Tool_testing\data\Site1\wetlands.shp"
img_dir = r"D:\3rdStudy_ONeil\wetland_identification\wetland_id_CNNs\tensorflow\models\research\deeplab\datasets\wetlands\dataset"
ImageSets = os.path.join(img_dir, "ImageSets")
JPEGImages = os.path.join(img_dir, "JPEGImages")
SegmentationClass = os.path.join(img_dir, "SegmentationClass")
TifTiles = os.path.join(img_dir, "TifTemp")
shp_dir = os.path.join(img_dir, "WetlandsSHP")
trainImg = os.path.join(ImageSets, 'train.txt')
trainvalImg = os.path.join(ImageSets, 'trainval.txt')
valImg = os.path.join(ImageSets, 'val.txt')
path_to_sh = r"D:\3rdStudy_ONeil\wetland_identification\wetland_id_CNNs\tensorflow\models\research\deeplab\datasets"

#set up directories
dirs = [ImageSets, JPEGImages, SegmentationClass, TifTiles, shp_dir]
for dir in dirs:
    if not os.path.exists(dir):
        os.mkdir(dir)

#params
train_percent = 0.6
trainval_percent = 0.2
img_tile_size = 512
train_iter = 500
train_crop_size = img_tile_size + 1

#============================================ create dataset of wetland images =========================================
# eligble_images = build_imgs(tif_in, shp_in, img_dir, tilesize=img_tile_size) #uncomment to rerun

#from eligible images, split into three datasets
# trainList, trainvalList, valList = \
#     split_imgs(eligble_images, trainImg, trainvalImg, valImg, train_percent, trainval_percent)

trainList = [line.rstrip('\n') for line in open(trainImg, "r")]
trainvalList = [line.rstrip('\n') for line in open(trainvalImg, "r")]
valList = [line.rstrip('\n') for line in open(valImg, "r")]

#=========================================  build wetland dataset as tf records ========================================
os.chdir(path_to_sh)
print (os.getcwd())
#build wetland dataset as tfrecords
cmd = "{}".format("convert_wetlands.sh")
# subprocess.call(cmd, shell=True)

#==============================================  train model on tfrecords  =============================================
#train_wetlands.sh takes 2 args: number of training iterations and train_crop_size
cmd = "{}".format("train_wetlands.sh {} {}".format(train_iter, train_crop_size))
subprocess.call(cmd, shell=True)

#==============================================  visualize model results  =============================================
#vis_wetlands.sh takes 1 arg: train_crop_size ('vis_crop_size')
cmd = "{}".format("vis_wetlands.sh {}".format(train_crop_size))
subprocess.call(cmd, shell=True)
