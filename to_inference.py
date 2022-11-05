from pickle import FALSE
from typing import List

import cv2
import numpy as np
import torch

from models.common import DetectMultiBackend
from utils.general import check_img_size,non_max_suppression,scale_coords, xyxy2xywh
from utils.torch_utils import select_device


class Inference(object):
    
    DEVIATION_X = 0
    DIRECTION = 0
    HIGH_EIGHT = 0
    LOW_EIGHT = 0
    # 目标位置
    TARGET_X = 0
    # 判断夹矿方式
    FLAG = 1

    def __init__(self,weights):
        # 加载模型
        self.device = select_device('cpu')
        self.model = DetectMultiBackend(weights, device=self.device)
        self.stride = self.model.stride 
        self.imgsz = check_img_size((320,320),s=self.stride)
        self.model.model.float()
    
    def radix_sort(arr:List[int]):
        n = len(str(max(arr)))  # 记录最大值的位数
        for k in range(n):#n轮排序
            # 每一轮生成10个列表
            bucket_list=[[] for i in range(10)]#因为每一位数字都是0~9，故建立10个桶
            for i in arr:
                # 按第k位放入到桶中
                bucket_list[i//(10**k)%10].append(i)
            # 按当前桶的顺序重排列表
            arr=[j for i in bucket_list for j in i]
        return arr

    def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)

    # 进行推理 绘制图像 结算出最优 发送数据
    def to_inference(self, frame, device, model, imgsz, stride,mode = 1, conf_thres=0.45, iou_thres=0.45):
        img_size = frame.shape
        img0 = frame 
        img = Inference.letterbox(img0,imgsz,stride=stride)[0]
        img = img.transpose((2,0,1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.

        # 每次初始化防止数据未刷新自己走，可能会慢一些
        Inference.DEVIATION_X = 0
        Inference.DIRECTION = 0
        Inference.HIGH_EIGHT = 0
        Inference.LOW_EIGHT = 0

        if len(img.shape) == 3:
            img = img[None]

        pred = model(img)
        pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)
        aims = []
        confs = []
        arr = []
        rects = []

        

        for i ,det in enumerate(pred): 
            gn = torch.tensor(img0.shape)[[1,0,1,0]]
            if len(det):
                det[:,:4] = scale_coords(img.shape[2:], det[:, :4],img0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1,4)) / gn).view(-1).tolist()
                    line = (cls, *xywh)
                    aim = ('%g ' * len(line)).rstrip() % line 
                    aim = aim.split(' ')
                    # 筛选出自信度大于70%
                    if float(conf) > 0.7:
                        aims.append(aim)
                        confs.append(float(conf))

            if len(aims):
                for i,det in enumerate(aims):
                    tag, x_center, y_center, width, height = det
                    x_center, width = float(x_center) * img_size[1], float(width) * img_size[1]
                    y_center, height = float(y_center) * img_size[0], float(height) * img_size[0]
                    top_left = (int(x_center - width * 0.5), int(y_center - height * 0.5))
                    top_right = (int(x_center + width * 0.5), int(y_center - height * 0.5))
                    bottom_left = (int(x_center - width * 0.5), int(y_center + height * 0.5))
                    bottom_right = (int(x_center + width * 0.5), int(y_center + height * 0.5))
                    

                    Inference.draw_inference(frame, top_left, top_right, bottom_right, tag, confs, i, mode)
                    tohsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    toinRange = cv2.inRange(tohsv, (0, 0, 236), (160, 89, 255))
                    contours, _ = cv2.findContours(toinRange, 0, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for i in range(0,len(contours)):
                        re = cv2.boundingRect(contours[i])
                        top_left_x, top_right_y, w, h = re
                        center_x, center_y = top_left_x + w / 2, top_right_y + h / 2
                        if center_x > top_left[0] and  center_y > top_left[1] and center_x < bottom_right[0] and center_y < bottom_right[1] :
                            rects.append(re)
                            # cv2.rectangle(frame,re,(255,0,255),3)

                    # 缺少roll轴旋转45度的情况 该怎么计算
                    top_left_point = Inference.near_compare(rects, top_left, 1)
                    bottom_left_point = Inference.near_compare(rects, bottom_left, 3)
                    bottom_right_point = Inference.near_compare(rects, bottom_right, 4)
                    print(top_left_point, bottom_left_point, bottom_right_point)
                    distance_level = Inference.compute_distance(bottom_left_point, bottom_right_point)
                    distance_vertical = Inference.compute_distance(top_left_point, bottom_left_point)
                    print(distance_vertical, distance_level)
                    if distance_vertical > distance_level:
                        pitch_angle = 0
                    else:
                        pitch_radians = np.arcsin(distance_vertical/distance_level)
                        pitch_angle = 90 - Inference.radians_to_angle(pitch_radians)
                    
                    if bottom_right_point[1] == bottom_left_point[1]:
                        roll_angle = 0
                    else:
                        k = (bottom_right_point[1] - bottom_left_point[1]) / (bottom_right_point[0] - bottom_left_point[0])
                        roll_radians = np.arctan(-k)
                        roll_angle = Inference.radians_to_angle(roll_radians)

                    print("pitch_angle: ", pitch_angle, "\n", "roll_angle: ", roll_angle)
    '''
    1 2
    3 4
    '''
    def near_compare(rects, point, mode):
        temp = 1280*800
        Point= []
        for i in range (0,len(rects)):
            if mode == 1: # 左上
                temp_point = [rects[i][0], rects[i][1]]
                distance = np.sqrt((point[0] - temp_point[0]) ** 2 + (point[1] - temp_point[1]) ** 2)
            elif mode == 2: # 右上
                temp_point = [rects[i][0] + rects[i][2], rects[i][1]]
                distance = np.sqrt((point[0] - temp_point[0] ) ** 2 + (point[1] - temp_point[1]) ** 2)
            elif mode == 3: # 左下
                temp_point = [rects[i][0], rects[i][1] + rects[i][3]]
                distance = np.sqrt((point[0] - temp_point[0]) ** 2 + (point[1] - temp_point[1]) ** 2)
            elif mode == 4: # 右下
                temp_point = [rects[i][0] + rects[i][2], rects[i][1] + rects[i][3]]
                distance = np.sqrt((point[0] - temp_point[0]) ** 2 + (point[1] - temp_point[1]) ** 2)
            if distance < temp:
                temp = distance
                Point.clear()       
                Point.append(temp_point[0])
                Point.append(temp_point[1])
                
        # print(Point)
        return Point
        
    def compute_distance(Point1, Point2):
        distance = np.sqrt((Point1[0] - Point2[0]) ** 2 + (Point1[1] - Point2[1]) ** 2)
        return distance

    def radians_to_angle(radians_value):
        PI = 3.14159265359
        angle = radians_value * 180 / PI
        return angle

    # 绘制推理框
    def draw_inference(frame, top_left, top_right, bottom_right, tag, confs, i, mode = 1):
        if mode == True:
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 255), 3, 8)
            cv2.putText(frame,str(float(round(confs[i], 2))), top_right, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(frame, tag, top_left, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 4)

    # 将数据显示出来
    def draw_data(frame, img_size, mode = 1):
        if mode == True:
            cv2.putText(frame, "judge_x = " + str(Inference.DEVIATION_X), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.line(frame, (Inference.TARGET_X, 0), (Inference.TARGET_X, int(img_size[0])), (255, 0, 255), 3)
            cv2.putText(frame, 'direction: ' + str(Inference.DIRECTION), (0, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(frame, 'high_eight: ' + str(Inference.HIGH_EIGHT), (0, 210), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(frame, 'low_eight: ' + str(Inference.LOW_EIGHT), (0, 260), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)