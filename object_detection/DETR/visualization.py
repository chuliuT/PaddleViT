#!/usr/bin/env python
# coding: utf-8

import sys
import os
import time
import logging
import argparse
import random
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
sys.path.append("./DETR")
from DETR.coco import build_coco
from DETR.coco import get_dataloader
from DETR.coco_eval import CocoEvaluator
from DETR.config import get_config
from DETR.config import update_config
from DETR.utils import WarmupCosineScheduler
from DETR.utils import AverageMeter,NestedTensor
from DETR.detr import build_detr
import DETR.transforms as T
from PIL import Image
from paddle.vision.transforms import Compose, Normalize,RandomResizedCrop,Resize,ToTensor
import matplotlib.pyplot as plt
import cv2
config = get_config(cfg_file="/home/aistudio/DETR/configs/detr_resnet50.yaml")
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]
model, criterion, postprocessors = build_detr(config)
model.set_state_dict(paddle.load("/home/aistudio/detr_resnet50.pdparams"))
model.eval()
print(model)

tfs=Compose([Resize(800),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


image_name="000000132635.jpg"
img=Image.open(image_name)
print(img.size)

t1=tfs(img).unsqueeze(0)

n,c,h,w=t1.shape

t2=NestedTensor(t1,paddle.zeros((1, h, w), dtype='int32'))

with paddle.no_grad():
    out=model(t2)

h,w=img.size
result=postprocessors["bbox"](out,paddle.to_tensor([[w,h]]))


COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
# plt.figure(figsize=(12,8))
img=Image.open(image_name)
image=np.array(img)

# ax = plt.gca()
for p, (xmin, ymin, xmax, ymax), lbl,c in zip(result[0]["scores"].numpy().tolist(), result[0]["boxes"].numpy().tolist(), result[0]["labels"].numpy().tolist(),COLORS * 100):
    cl = p
    if p>0.5:
        print((int(xmin),int(ymin)),(int(xmax),int(ymax)))
        cv2.rectangle(image,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,255,0),2)
        text = f'{p:0.2f}'
        cv2.putText(image,text+"|"+CLASSES[lbl],(int(xmin),int(ymin)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)
cv2.imwrite("result.jpg",image[:,:,::-1])
# plt.show()





