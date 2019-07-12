import os 

"""
Train a model with images from all sites, but limit site contributions so they are
all equal (all sites only contribute 9 images)
"""
run_name = "run_G34"
site_name = "Site4"

train_thresh = None

img_dir = os.path.join(r"D:\3rdStudy_ONeil\wetland_id\CNNs\pytorch\dataset", site_name) 
ImageSets = os.path.join(img_dir, "ImageSets") 
IMG_Tiles = os.path.join(img_dir, "Imgs") 
GT_Tiles = os.path.join(img_dir, "Verif") 
train_percent = 0.7 
img_tile_size = 320  
nd_val = 0.


base_dset = r"D:\3rdStudy_ONeil\wetland_id\CNNs\pytorch\dataset\{}\ImageSets"

train_set1 = os.path.join(base_dset.format("Site1"), "train.txt")
test_set1 = os.path.join(base_dset.format("Site1"), "val.txt")

train_set2 = os.path.join(base_dset.format("Site2"), "train.txt")
test_set2 = os.path.join(base_dset.format("Site2"), "val.txt")

train_set3 = os.path.join(base_dset.format("Site3"), "train.txt")
test_set3 = os.path.join(base_dset.format("Site3"), "val.txt")

train_set4 = os.path.join(base_dset.format("Site4"), "train.txt")
test_set4 = os.path.join(base_dset.format("Site4"), "val.txt")

"""CHANGE THIS"""
train_sets = [train_set4]#, train_set2, train_set3, train_set4]
test_sets = [test_set4]#, test_set2, test_set3, test_set4]


#deepnets params
WINDOW_SIZE = (64,64) # Patch size, orig 256 
STRIDE = 8 # Stride for testing
IN_CHANNELS = 4 # Number of input channels (e.g. RGB)
BATCH_SIZE = 24 # Number of samples in a mini-batch
st = 1 #step
w = 5 #window size
base_lr = 0.01
MOMENTUM=0.9
WEIGHT_DECAY = 0.0005
epochs = 100