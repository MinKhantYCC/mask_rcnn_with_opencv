import os
import cv2
from utils import get_detections
import numpy as np

# define paths
cfg_pth = './model/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt.txt'
weight_path = './model/frozen_inference_graph.pb'
class_names_path = './model/mscoco_labels.names'
class_names = []
with open(class_names_path, 'r', newline='\n') as f:
    texts = f.read()
class_names = texts.split('\n')
# print(class_names)

img_path = 'jp_street.jpg'

# load image
img = cv2.imread(img_path)
H, W, C = img.shape

# load model
net = cv2.dnn.readNet(model=weight_path, config=cfg_pth)

# convert image
blob = cv2.dnn.blobFromImage(img)

# get mask
boxes, masks = get_detections(net, blob)
# print(boxes)
colors = [(np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255),) for j in range(90)]

# draw mask
empty_img = np.zeros((H,W, C))
for j in range(len(masks)):
    box = boxes[0, 0, j]
    score = box[2]
    # print(score)
    if score > 0.5:
        mask = masks[j]
        class_id = box[1]
        x1, y1, x2, y2 = int(box[3]*W), int(box[4]*H), int(box[5]*W), int(box[6]*H)
        # img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)
        cv2.putText(img, class_names[int(class_id)], org=(x1+20,y1+20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, thickness=3,
                    color= (255,0,0),
                    )
        mask = mask[int(class_id)]
        mask = cv2.resize(mask, (x2-x1, y2-y1))
        _, mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
        for c in range(3):
            empty_img[y1:y2, x1:x2, c] = mask * colors[int(class_id)][c]

overlay = ((0.6*empty_img)+(0.4*img)).astype("uint8")

# cv2.imshow("mask", mask)
# cv2.imshow("img", img)
cv2.imshow("Segmentized", overlay)
cv2.imwrite(f"outputs_2.png", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()