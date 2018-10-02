import os
import sys
from random import shuffle

flags = dict(n = '', b = '', e='', s='')
options = dict(training_dataset=r"D:\3rdStudy_ONeil\data\s1_scratch\imgs",
               model='',
               classes=['wetland'],
               name='S1_test',
               logs=r"D:\3rdStudy_ONeil\data\s1_scratch\logs",
               epochs=200,
               steps_per_epoch=3000,
               rois_per_image=64,
               images_per_gpu=1,
               gpu_count=1,
               mini_mask_size=[],
               validation_steps=100,
               images_max_dim=1280,
               images_min_dim=256,
               backbone='resnet101')


def main(options, flags):

    from config import ModelConfig
    import utils
    import model as modellib

    try:
        dataset = options['training_dataset']
        initialWeights = options['model']
        classes = options['classes']
        name = options['name']
        logs = options['logs']
        epochs = int(options['epochs'])
        stepsPerEpoch = int(options['steps_per_epoch'])
        ROIsPerImage = int(options['rois_per_image'])
        imagesPerGPU = int(options['images_per_gpu'])
        GPUcount = int(options['gpu_count'])
        miniMaskSize = options['mini_mask_size']
        validationSteps = int(options['validation_steps'])
        imMaxDim = int(options['images_max_dim'])
        imMinDim = int(options['images_min_dim'])
        backbone = options['backbone']
    except KeyError:
        dataset = options[b'training_dataset'].decode('utf-8')
        initialWeights = options[b'model'].decode('utf-8')
        classes = options[b'classes'].decode('utf-8').split(',')
        name = options[b'name'].decode('utf-8')
        logs = options[b'logs'].decode('utf-8')
        epochs = int(options[b'epochs'])
        stepsPerEpoch = int(options[b'steps_per_epoch'])
        ROIsPerImage = int(options[b'rois_per_image'])
        imagesPerGPU = int(options[b'images_per_gpu'])
        GPUcount = int(options[b'gpu_count'])
        miniMaskSize = options[b'mini_mask_size'].decode('utf-8')
        validationSteps = int(options[b'validation_steps'])
        imMaxDim = int(options[b'images_max_dim'])
        imMinDim = int(options[b'images_min_dim'])
        backbone = options[b'backbone'].decode('utf-8')

        newFlags = dict()
        for flag, value in flags.items():
            newFlags.update({flag.decode('utf-8'): value})
        flags = newFlags

    if not flags['b']:
        trainBatchNorm = False
    else:
        # None means train in normal mode but do not force it when inferencing
        trainBatchNorm = None

    if not flags['n']:
        print("resizing")
        # Resize and pad with zeros to get a square image of
        # size [max_dim, max_dim].
        resizeMode = 'square'
    else:
        print("no resizing")
        resizeMode = 'none'

    # Configurations
    config = ModelConfig(name=name,
                         imagesPerGPU=imagesPerGPU,
                         GPUcount=GPUcount,
                         numClasses=len(classes) + 1,
                         trainROIsPerImage=ROIsPerImage,
                         stepsPerEpoch=stepsPerEpoch,
                         miniMaskShape=miniMaskSize,
                         validationSteps=validationSteps,
                         imageMaxDim=imMaxDim,
                         imageMinDim=imMinDim,
                         backbone=backbone,
                         trainBatchNorm=trainBatchNorm,
                         resizeMode=resizeMode)
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=logs)


    # Load weights
    if initialWeights:
        print("Loading weights {}".format(initialWeights))
    if initialWeights and flags['e']:
        model.load_weights(initialWeights, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif initialWeights:
        model.load_weights(initialWeights, by_name=True)

    print('Reading images from dataset {}'.format(dataset))
    images = list()
    for root, subdirs, _ in os.walk(dataset):
        if not subdirs:
            images.append(root) #this should be every img # in directory
    shuffle(images)

    if flags['s']:
        # Write list of unused images to logs
        testImagesThreshold = int(len(images) * .9)
        print('List of unused images saved in the logs directory'
                        'as "unused.txt"')
        with open(os.path.join(logs, 'unused.txt'), 'w') as unused:
            for filename in images[testImagesThreshold:]:
                unused.write('{}\n'.format(filename))
    else:
        testImagesThreshold = len(images)

    evalImagesThreshold = int(testImagesThreshold * .75)

    # augmentation = imgaug/augmenters.Fliplr(0.5)

    # Training dataset
    trainImages = images[:evalImagesThreshold]
    dataset_train = utils.Dataset()
    dataset_train.import_contents(classes, trainImages, name)
    dataset_train.prepare()

    # Validation dataset
    evalImages = images[evalImagesThreshold:testImagesThreshold]
    dataset_val = utils.Dataset()
    dataset_val.import_contents(classes, evalImages, name)
    dataset_val.prepare()

    if initialWeights:
        # Training - Stage 1
        # Adjust epochs and layers as needed
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=int(epochs / 7),
                    layers='heads')  # augmentation=augmentation

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        # divide the learning rate by 10 if ran out of memory or
        # if weights exploded
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=int(epochs / 7) * 3,
                    layers='4+')  # augmentation=augmentation

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        # out of if statement
    else:
        print("Training all layers")
        # out of if statement

    # divide the learning rate by 100 if ran out of memory or
    # if weights exploded
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=epochs,
                layers='all')  # augmentation=augmentation


if __name__ == "__main__":
    main(options, flags)
