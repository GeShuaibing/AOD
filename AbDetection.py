import numpy as np
import cv2 as cv


class Cur_Frame_box():
    def __init__(self, label, xyxy, id):
        super(Cur_Frame_box, self).__init__()
        self.id = id  #追踪id
        self.label = label
        self.box = xyxy
        self.status = None  # 运动，静止
        self.time = 0  # 静止持续时间
        self.distance = 0
        self.direction = (0, 0)


def curFb_list(cls, xyxy, names, ids):
    current = []
    for i in range(len(cls)):
        label = names[cls[i]]
        box = xyxy[i]
        current.append(Cur_Frame_box(label, box, int(ids[i])))
    return current


def box_iou_xyxy(box1, box2):
    # 获取box1左上角和右下角的坐标
    x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
    # 计算box1的面积
    s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
    # 获取box2左上角和右下角的坐标
    x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]
    # 计算box2的面积
    s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)

    # 计算相交矩形框的坐标
    xmin = np.maximum(x1min, x2min)
    ymin = np.maximum(y1min, y2min)
    xmax = np.minimum(x1max, x2max)
    ymax = np.minimum(y1max, y2max)
    # 计算相交矩形行的高度、宽度、面积
    inter_h = np.maximum(ymax - ymin + 1., 0.)
    inter_w = np.maximum(xmax - xmin + 1., 0.)
    intersection = inter_h * inter_w
    # 计算相并面积
    union = s1 + s2 - intersection
    # 计算交并比
    iou = intersection / union
    return iou


def abdetect(last, current):
    # if len(current) == 0:
    #     return current
    # elif len(last) == 0:
    #     return current
    # else:
    # print("abdetect0")
    for x in last:
        for y in current:
            # if x.label == y.label:
            if x.id == y.id:
                iou = box_iou_xyxy(x.box, y.box)
                # print(iou)
                if iou > 0.9:
                    y.time = x.time + 1
    return current


def draw_lost(current, img):

    for y in current:
        if y.time > 40 and y.label in ('backpack','handbag','suitcase'):
            cv.putText(img, "!", (int((y.box[2] + y.box[0]) / 2), int((y.box[1] + y.box[3]) / 2)),
                       cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3,
                       cv.LINE_AA)
            cv.putText(img, 'existing lost!', (25, 50), cv.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 1)
    return img
