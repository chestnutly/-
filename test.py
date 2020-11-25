# -*- coding: utf-8 -*-
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import time
from mrcnn.config import Config
from datetime import datetime
# Root directory of the project
start = time.clock()
ROOT_DIR = os.getcwd()
os.environ['KERAS_BACKEND']='tensorflow'
 
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
# sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
# from samples.coco import coco
 
 
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
 
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(MODEL_DIR ,"shapes20201124T1731/mask_rcnn_shapes_0050.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    print("cuiwei***********************")
 
# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "test-image")
 
class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"
 
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
 
    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 3 shapes
 
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 384
 
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels
 
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE =100
 
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100
 
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50
 
#import train_tongue
#class InferenceConfig(coco.CocoConfig):
class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
config = InferenceConfig()
 
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
 
 
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
 
# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)
 
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'chahua','pengao','tufeng']


#读取多个图片程序
'''

# Load a random image from the images folder
right={'21.jpg':'pengao','26.jpg':'chahua','31.jpg':'chahua','36.jpg':'chahua','43.jpg':'chahua','51.jpg':'pengao','64.jpg':'tufeng','68.jpg':'chahua','78.jpg':'chahua','128.jpg':'tufeng','146.jpg':'tufeng','162.jpg':'chahua','173.jpg':'chahua','176.jpg':'pengao','208.jpg':'tufeng','211.jpg':'tufeng','369.jpg':'chahua','415.jpg':'chahua'}
a=datetime.now()
file_names = next(os.walk(IMAGE_DIR))[2]
#print(file_names)
#image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
save=[]
right_count,wrong_count=0,0
#image = skimage.io.imread("./images/5951960966_d4e1cda5d0_z.jpg")     #直接单个读取
for j in range(len(file_names)):
    image = skimage.io.imread(os.path.join(IMAGE_DIR, file_names[j]))
# Run detection
    results = model.detect([image], verbose=1)
# Visualize results
#print("检测运行时间:")
#print("秒",(b-a).seconds)
    print(file_names[j])
    r = results[0]
    for i in range(len(r['class_ids'])):
        print(class_names[int(r['class_ids'][i])])
        save.append(class_names[int(r['class_ids'][i])])
        if right[file_names[j]]==str(class_names[int(r['class_ids'][i])]):
            print(True)
            right_count+=1
        else:
            print(False)
            wrong_count+=1
    save.append('ok')
b=datetime.now()
print(save)
print("正确率:")
print(right_count/(right_count+wrong_count))
'''


#单个读取照片程序
image = skimage.io.imread("./test-image/21.jpg")     #直接单个读取
results = model.detect([image], verbose=1)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])