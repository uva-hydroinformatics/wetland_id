# imports and stuff
import numpy as np
from skimage import io
from tqdm import tqdm_notebook as tqdm
import sklearn.metrics as skmetrics
import random
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data #as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from torch.autograd import Variable
import sys
import os
import pandas as pd
import re
from io import StringIO
from datetime import datetime
import params

DATA_FOLDER = r"D:\3rdStudy_ONeil\wetland_id\CNNs\pytorch\dataset\{}\Imgs"
LABEL_FOLDER = r"D:\3rdStudy_ONeil\wetland_id\CNNs\pytorch\dataset\{}\Verif"

# Parameters
WINDOW_SIZE = params.WINDOW_SIZE
STRIDE = params.STRIDE
IN_CHANNELS = params.IN_CHANNELS
BATCH_SIZE = params.BATCH_SIZE
st = params.st
w = params.w


LABELS = ["No_Data", "nonwetland", "wetland"] # Label names, correspond to values 0, 1, 2
N_CLASSES = len(LABELS) # Number of classes
WEIGHTS = torch.tensor([0.02, 0.08, 0.9])
CACHE = True # Store the dataset in-memory
	
palette = {0 : (0, 0, 0), # nodata (black)
           1 : (51, 102, 0),    # nonwetland (green)
           2 : (0, 204, 204)}     # wetland (teal)

invert_palette = {v: k for k, v in palette.items()}

def convert_to_color(arr_2d, palette=palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d

# Utils
def _get_random_pos(img_shape, window_shape):
    w, h = window_shape
    W, H = img_shape
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2

def get_random_pos(img_shape, window_shape, mask=None):
    """ Extract of 2D random patch of shape window_shape in the image """
    if mask is None:
        return _get_random_pos(img_shape, window_shape)
    else:
        x1, x2, y1, y2 = _get_random_pos(img_shape, window_shape)
        w,h = window_shape
        while np.count_nonzero(mask[x1:x2,y1:y2]) < 0.8 * mask[x1:x2,y1:y2].size:
            x1, x2, y1, y2 = _get_random_pos(img_shape, window_shape)
    return x1, x2, y1, y2

def CrossEntropy2d(input, target, weight=None, size_average=True):
    """ 2D version of the cross entropy loss """
    dim = input.dim()
    
    if dim == 2:
        return F.cross_entropy(input, target, weight, size_average)#, ignore_index=3)
    elif dim == 4:
        output = input.view(input.size(0),input.size(1), -1)
        output = torch.transpose(output,1,2).contiguous()
        output = output.view(-1,output.size(2))
        target = target.view(-1)
        return F.cross_entropy(output, target,weight, size_average)#, ignore_index=3)
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))

def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size

def sliding_window(top, step=st, window_size=(w,w)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]
            
def count_sliding_window(top, step=st, window_size=(w,w)):
    """ Count the number of windows in an image """
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c

def grouper(n, iterable):
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def metrics(predictions, gts, label_values=LABELS):
    cm = skmetrics.confusion_matrix(
            gts,
            predictions,
            range(len(label_values)))
    
    print("Confusion matrix :")
    print(cm)
    print("---")
    
    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("{} pixels processed".format(total))
    print("Total accuracy : {}%".format(accuracy))
    print("---")
    
    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i,i] / (np.sum(cm[i,:]) + np.sum(cm[:,i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("{}: {}".format(label_values[l_id], score))
    print("---")
        
    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total*total)
    kappa = (pa - pe) / (1 - pe);
    print("Kappa: " + str(kappa)) 
    
    #report
    print(skmetrics.classification_report(gts, predictions))
    precision = skmetrics.precision_score(gts, predictions, labels = [2], pos_label=2, average=None)
    recall = skmetrics.recall_score(gts, predictions, labels=[2], pos_label=2, average=None)
    
    return precision[0], recall[0]
	
def overall_metrics(predictions, gts, label_values=LABELS):
    
    #Confusion Matrix
    cm = skmetrics.confusion_matrix(
            gts,
            predictions,
            range(len(label_values)))
    
    cm_pd= pd.DataFrame(cm, index=['GT NoData','GT Nonwetlands', 'GT Wetlands'], 
                     columns=['P NoData','P Nonwetlands', 'P Wetlands']).round(4)
    cm_pd.loc[u'Σ'] = cm_pd.sum()
    cm_pd[u'Σ'] = cm_pd.sum(axis=0)
    cm_pd[u'Σ'] = cm_pd.sum(axis=1)
    
    #accuracy score
    total = sum(sum(cm))
    acc_old = sum([cm[x][x] for x in range(len(cm))])
    acc_old *= 100 / float(total)
    acc_old_pd = pd.Series(acc_old, index=['Old Method Accuracy:'])
    accuracy = skmetrics.accuracy_score(gts, predictions)
    acc_pd = pd.Series(accuracy, index=['Accuracy score:'])
    acc_pd = acc_pd.round(4)
    acc_scores_all= acc_pd.append([acc_old_pd])
    
    #IoU
    iou = skmetrics.jaccard_score(gts, predictions, average=None)
    iou_pd = pd.Series(iou, index=['ND IoU:','NW IoU:','W IoU:'])
    
    
    #report
    class_report = skmetrics.classification_report(gts, predictions)
    class_report = re.sub(r" +", " ", class_report).replace("avg / total", "avg/total").replace("\n ", "\n")
    class_report_df = pd.read_csv(StringIO("Classes" + class_report), sep=' ', index_col=0, error_bad_lines=False) 
    
    return cm_pd, class_report_df, acc_scores_all, iou_pd

# Dataset class        
class WETLANDS_dataset(torch.utils.data.Dataset):
    def __init__(self, ids, cache=CACHE, augmentation=True):
        super(WETLANDS_dataset, self).__init__()
        
        self.augmentation = augmentation
        self.cache = cache
        
        #parse site names and imgIDs
        site_names = [ i.split('_')[0] for i in ids ]
        img_ids = [ i.split('_')[1] for i in ids ]
        
        self.data_files = [os.path.join(DATA_FOLDER.format(site_names[x]), img_ids[x] + ".tif") for x in range(len(img_ids))]
        self.label_files = [os.path.join(LABEL_FOLDER.format(site_names[x]), img_ids[x] + ".tif") for x in range(len(img_ids))]
    
        
        # Sanity check : raise an error if some files do not exist
        for f in self.data_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))
        
        # Initialize cache dicts
        self.data_cache_ = {}
        self.label_cache_ = {}
            
    
    def __len__(self):
        # Default epoch size is 10 000 samples
        return 10000
    
    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True
        
        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))
            
        return tuple(results)

    def __getitem__(self, i):
        # Pick a random image
        random_idx = random.randint(0, len(self.data_files) - 1)
        
        # If the tile hasn't been loaded yet, put in cache
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            # Data is normalized in [0, 1]
            arr = io.imread(self.data_files[random_idx])
            data = arr.transpose((2,0,1))
            if self.cache:
                self.data_cache_[random_idx] = data
            
        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else: 
            # Labels are converted from RGB to their numeric values
            label = np.asarray((io.imread(self.label_files[random_idx])), dtype='int64')
            if self.cache:
                self.label_cache_[random_idx] = label

        # Get a random patch
        x1, x2, y1, y2 = _get_random_pos(label.shape, WINDOW_SIZE)
        data_p = data[:, x1:x2,y1:y2]
        label_p = label[x1:x2,y1:y2]
        
        # Data augmentation
        data_p, label_p = self.data_augmentation(data_p, label_p)
       
        # Return the torch.Tensor values
        return (torch.from_numpy(data_p),
                torch.from_numpy(label_p))
        
class SegNet(nn.Module):
    # SegNet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)
    
    def __init__(self, in_channels=IN_CHANNELS, out_channels=N_CLASSES):
        super(SegNet, self).__init__()
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)
        
        self.conv1_1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(64)
        
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(128)
        
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(256)
        
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(512)
        
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(512)
        
        self.conv5_3_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_D_bn = nn.BatchNorm2d(512)
        self.conv5_2_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_D_bn = nn.BatchNorm2d(512)
        self.conv5_1_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_D_bn = nn.BatchNorm2d(512)
        
        self.conv4_3_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_D_bn = nn.BatchNorm2d(512)
        self.conv4_2_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_D_bn = nn.BatchNorm2d(512)
        self.conv4_1_D = nn.Conv2d(512, 256, 3, padding=1)
        self.conv4_1_D_bn = nn.BatchNorm2d(256)
        
        self.conv3_3_D = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_D_bn = nn.BatchNorm2d(256)
        self.conv3_2_D = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(256)
        self.conv3_1_D = nn.Conv2d(256, 128, 3, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(128)
        
        self.conv2_2_D = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(128)
        self.conv2_1_D = nn.Conv2d(128, 64, 3, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(64)
        
        self.conv1_2_D = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(64)
        self.conv1_1_D = nn.Conv2d(64, out_channels, 3, padding=1)
        
        self.apply(self.weight_init)
        
    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x.float())))
        x = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x, mask1 = self.pool(x)
        
        # Encoder block 2
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        x = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x, mask2 = self.pool(x)
        
        # Encoder block 3
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x, mask3 = self.pool(x)
        
        # Encoder block 4
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        x = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x, mask4 = self.pool(x)
        
        # Encoder block 5
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        x = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        x, mask5 = self.pool(x)
        
        # Decoder block 5
        x = self.unpool(x, mask5)
        x = self.conv5_3_D_bn(F.relu(self.conv5_3_D(x)))
        x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))
        
        # Decoder block 4
        x = self.unpool(x, mask4)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))
        
        # Decoder block 3
        x = self.unpool(x, mask3)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))
        
        # Decoder block 2
        x = self.unpool(x, mask2)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))
        
        # Decoder block 1
        x = self.unpool(x, mask1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        x = F.log_softmax(self.conv1_1_D(x))
        return x
		
    
def init_model(base_lr = params.base_lr, MOMENTUM = params.MOMENTUM, WEIGHT_DECAY = params.WEIGHT_DECAY):

    # instantiate the network
    net = SegNet()
    net.cuda()

    params_dict = dict(net.named_parameters())
    params = []
    for key, value in params_dict.items():
        if '_D' in key:
            # Decoder weights are trained at the nominal learning rate
            params += [{'params':[value],'lr': base_lr}]
        else:
            # Encoder weights are trained at lr / 2 (we have VGG-16 weights as initialization)
            params += [{'params':[value],'lr': base_lr / 2}]
    
    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    
    # We define the scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)
    
    return net, scheduler, optimizer

def load_data(train_ids):
    
    print("Tiles for training : {}\n".format(train_ids))
    
    train_set = WETLANDS_dataset(train_ids)
    print(train_set.data_files)
    print(train_set.label_files)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)
    return train_loader


def test(net, test_ids, all=False, stride=WINDOW_SIZE[0]//2, batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    print("Tiles for testing : {}\n".format(test_ids))
    # Use the network on the test set
    
    #parse site names and imgIDs
    site_names = [ i.split('_')[0] for i in test_ids ]
    img_ids = [ i.split('_')[1] for i in test_ids ]
    
    test_files = [os.path.join(DATA_FOLDER.format(site_names[x]), img_ids[x] + ".tif") for x in range(len(img_ids))]
    label_files = [os.path.join(LABEL_FOLDER.format(site_names[x]), img_ids[x] + ".tif") for x in range(len(img_ids))]
    
    test_images_list = []
    test_labels_list = []
    for t in range(len(test_ids)):
        arr = io.imread(test_files[t])
        test_images_list.append(arr)
        test_labels_list.append(np.asarray(io.imread(label_files[t]), dtype='int8'))
        
    test_images = tuple(test_images_list)
    test_labels = tuple(test_labels_list)

    all_preds = []
    all_gts = []
    
    # Switch the network to inference mode
    net.eval()

    for img, gt in tqdm(zip(test_images, test_labels), total=len(test_ids), leave=False):
        pred = np.zeros(img.shape[:2] + (N_CLASSES,))

        total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
        for i, coords in enumerate(tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total, leave=False)):
            # Display in progress results
            if i > 0 and total > 10 and i % int(10 * total / 100) == 0:
                    _pred = np.argmax(pred, axis=-1)
                    fig = plt.figure()
                    fig.add_subplot(1,3,1)
                    plt.title('RGB')
                    plt.imshow(np.asarray(255 * img, dtype='int8'))
                    fig.add_subplot(1,3,2)
                    plt.title('Prediction')
                    plt.imshow(convert_to_color(_pred))
                    fig.add_subplot(1,3,3)
                    plt.title('Ground Truth')
                    plt.imshow(convert_to_color(gt))
                    plt.show()
                    plt.close()
                    
            # Build the tensor
            image_patches = [np.copy(img[x:x+w, y:y+h]).transpose((2,0,1)) for x,y,w,h in coords]
            image_patches = np.asarray(image_patches)
            image_patches = Variable(torch.from_numpy(image_patches).cuda())#.no_grad() #change according to github issue
            # Do the inference
            outs = net(image_patches)
            outs = outs.data.cpu().numpy()
            
            # Fill in the results array
            for out, (x, y, w, h) in zip(outs, coords):
                out = out.transpose((1,2,0))
                pred[x:x+w, y:y+h] += out
            del(outs)

        pred = np.argmax(pred, axis=-1)

        # Display the result
        fig = plt.figure()
        fig.add_subplot(1,3,1)
        plt.title('RGB')
        plt.imshow(np.asarray(255 * img, dtype='uint8'))
        fig.add_subplot(1,3,2)
        plt.title('Prediction')
        plt.imshow(convert_to_color(pred))
        fig.add_subplot(1,3,3)
        plt.title('Ground Truth')
        plt.imshow(gt)
        plt.show()
        plt.close()

        all_preds.append(pred)
        all_gts.append(gt)

        # Compute some metrics
        metrics(pred.ravel(), gt.ravel())
        precision, recall = metrics(np.concatenate([p.ravel() for p in all_preds]), \
                                    np.concatenate([p.ravel() for p in all_gts]).ravel())
    
    #final_accuracy
    cm_pd, class_report_df, acc_scores_all, iou_pd = \
    overall_metrics(np.concatenate([p.ravel() for p in all_preds]), np.concatenate([p.ravel() for p in all_gts]).ravel())
    
    if all:
        return accuracy, all_preds, all_gts, cm_pd, class_report_df, acc_scores_all, iou_pd
    else:
        return precision, recall
		
def train(net, optimizer, epochs, train_loader, test_ids, savedir, run, scheduler=None, weights=WEIGHTS, save_epoch = 20):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()
    iter_ = 0
    start = datetime.now()
    
    for e in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            output = net(data)
            loss = F.cross_entropy(output, target, weight=weights)
            loss.backward()
            optimizer.step()

            losses[iter_] = loss.item() #change according to github issue
            mean_losses[iter_] = np.mean(losses[max(0,iter_-100):iter_])
            
            if iter_ % 100 == 0:
                rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0],(1,2,0)), dtype='uint8')
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.item(), accuracy(pred, gt)))
                plt.plot(mean_losses[:iter_]) and plt.show()
                plt.close()
                fig = plt.figure()
                fig.add_subplot(131)
                plt.imshow(rgb)
                plt.title('RGB')
                fig.add_subplot(132)
                plt.imshow(convert_to_color(gt))
                plt.title('Ground truth')
                fig.add_subplot(133)
                plt.title('Prediction')
                plt.imshow(convert_to_color(pred))
                plt.show()
                plt.close()
            iter_ += 1
            
            del(data, target, loss)
            
        if e % save_epoch == 0:
            # We validate with the largest possible stride for faster computing
            #epoch saves the accuracy achieved on the TESTING set achieved at each save epoch
            p, r = test(net, test_ids, all=False, stride=min(WINDOW_SIZE))
            torch.save(net.state_dict(), os.path.join(savedir, '{}_epoch{}_p{}_r{}'.format(run, e, np.round(p, 3), np.round(r, 3))))
    
    end = datetime.now()
    final_model = os.path.join(savedir, 'segnet_final')
    torch.save(net.state_dict(), final_model)
    print("Training complete, time elapsed: {} \nFinal model saved to: {}".format\
          ((end-start), final_model))
    return final_model                    