# 安装 OpenCV（如果 Kaggle 环境缺少）
# !pip install opencv-python-headless --quiet
#  https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg
# https://github.com/AlexeyAB/darknet/blob/master/data/coco.names
# https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 路径修改为你 Dataset 的路径
cfg = '/kaggle/input/yolov4/yolov4.cfg'
weights = '/kaggle/input/yolov4/yolov4.weights'
names = '/kaggle/input/yolov4/coco.names'
img_path = '/kaggle/input/test-image/dog.jpg'

# 加载类别标签
with open(names, 'r') as f:
    classes = [c.strip() for c in f]

# 加载 DNN 模型并设为 CPU 模式
net = cv2.dnn.readNetFromDarknet(cfg, weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# 读取和处理图像
img = cv2.imread(img_path)
h, w = img.shape[:2]
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416,416), swapRB=True, crop=False)
net.setInput(blob)

# 前向推理
outs = net.forward(net.getUnconnectedOutLayersNames())

# 提取检测框
conf_thr, nms_thr = 0.5, 0.4
boxes, confidences, class_ids = [], [], []
for out in outs:
    for det in out:
        scores = det[5:]
        cid = int(np.argmax(scores))
        conf = float(scores[cid])
        if conf > conf_thr:
            cx, cy, ww, hh = (det[0:4] * [w,h,w,h]).astype(int)
            x,y = int(cx-ww/2), int(cy-hh/2)
            boxes.append([x,y,ww,hh])
            confidences.append(conf)
            class_ids.append(cid)

idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thr, nms_thr)

# 绘制检测结果
for i in idxs.flatten():
    x,y,ww,hh = boxes[i]
    label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
    cv2.rectangle(img, (x,y), (x+ww,y+hh), (0,255,0), 2)
    cv2.putText(img, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)

# 显示最终图像
plt.figure(figsize=(8,8))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
