from pickle import FALSE
from typing import List

import cv2
import numpy as np
import torch
import video_capture

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

        contours = Inference.find_light(frame)

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
                    if float(conf) > 0.65:
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
                    # print(top_left, top_right, bottom_left, bottom_right)
                    
                    if tag == '0':                        
                        Inference.draw_inference(frame, top_left, top_right, bottom_right, tag, confs, i, mode)
                        arr.append(int(x_center - Inference.TARGET_X)) 
                    elif tag == '1':      
                        pass          

                if  tag == '0':
                    if abs(Inference.radix_sort(arr)[0]) < abs(Inference.radix_sort(arr)[len(arr)-1]):
                        Inference.DEVIATION_X = Inference.radix_sort(arr)[0]
                    else:
                        Inference.DEVIATION_X = Inference.radix_sort(arr)[len(arr)-1]

                    if mode == True:
                        cv2.putText(frame, "real_x = " + str(Inference.DEVIATION_X), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                    Inference.HIGH_EIGHT = (abs(Inference.DEVIATION_X) >> 8) & 0xff
                    Inference.LOW_EIGHT = abs(Inference.DEVIATION_X)  & 0xff
                    
                    if abs(Inference.DEVIATION_X ) < 24:
                        Inference.DEVIATION_X  = 0
                    if Inference.DEVIATION_X > 0:
                        Inference.DIRECTION = 1
                    Inference.draw_data(frame, img_size, mode)
                elif tag == '1':
                    target = 0
                    for i in range (0,len(aims)):
                        temp = 0
                        if float(aims[i][3]) * float(aims[i][4]) > temp:
                            temp = float(aims[i][3]) * float(aims[i][4])
                            target = i
                    tag, x_center, y_center, width, height = aims[target]
                    x_center, width = float(x_center) * img_size[1], float(width) * img_size[1]
                    y_center, height = float(y_center) * img_size[0], float(height) * img_size[0]
                    top_left = (int(x_center - width * 0.5), int(y_center - height * 0.5))
                    top_right = (int(x_center + width * 0.5), int(y_center - height * 0.5))
                    bottom_left = (int(x_center - width * 0.5), int(y_center + height * 0.5))
                    bottom_right = (int(x_center + width * 0.5), int(y_center + height * 0.5))
                    Inference.draw_inference(frame, top_left, top_right, bottom_right, tag, confs, i, mode)
                    for i in range(0,len(contours)):
                            rect = cv2.boundingRect(contours[i])
                            top_left_x, top_left_y, width, height = rect
                            light_center = top_left_x + width / 2, top_left_y + height / 2
                            # 筛选灯条  在指定区域内
                            if light_center[0] > top_left[0] and  light_center[1] > top_left[1] and light_center[0] < bottom_right[0] and light_center[1] < bottom_right[1] :
                                rects.append(rect)
                                # cv2.rectangle(frame,rect,(255,0,255),3)
                    try:
                        rects = Inference.area_compare(rects, frame)
                        for i in range (0,len(rects)):
                            cv2.rectangle(frame,rects[i],(255,0,255),3)
                    except:
                        print("No Find Light")
                    try:
                        top_left_point = Inference.near_compare(rects, top_left, 1)
                        top_right_point = Inference.near_compare(rects, top_right, 2)
                        bottom_left_point = Inference.near_compare(rects, bottom_left, 3)
                        bottom_right_point = Inference.near_compare(rects, bottom_right, 4)
                        print(top_left_point, top_right_point, bottom_left_point, bottom_right_point)                        
                        
                        # 均值处理角度 
                        distance_level_borrom = Inference.compute_distance(bottom_left_point, bottom_right_point)                        
                        distance_vertical_left = Inference.compute_distance(top_left_point, bottom_left_point)
                        distance_level_top = Inference.compute_distance(top_left_point, top_right_point)
                        distance_vertical_right = Inference.compute_distance(top_right_point, bottom_right_point)                        
                    except:
                        print("Analysis Point Error")                        
                    try:   
                        pitch_angle =Inference.compute_pitch(distance_level_top, distance_level_borrom, distance_vertical_left, distance_vertical_right, 31)  
                        roll_angle = Inference.compute_roll(top_left_point, top_right_point, bottom_left_point, bottom_right_point)
                        # 13度是误差极限 还是位置极佳准确的时候                    
                        print("pitch_angle: ", pitch_angle, "\n", "roll_angle: ", roll_angle)
                    except:
                        print("Analysis Angle Error")
    
    # 只有蓝色的没有红色的，得思考一下怎么加
    def find_light(frame):
        tohsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # toinRange = cv2.inRange(tohsv, (0, 0, 236), (160, 89, 255))
        toinRange = cv2.inRange(tohsv, (0, 0, 214), (109, 255, 255))
        contours, _ = cv2.findContours(toinRange, 0, cv2.CHAIN_APPROX_SIMPLE)  
        return contours

    # 面积筛选 排除roll旋转小灯条的干扰
    def area_compare(rects, frame):
        area_lists = []
        new_rects = []
        for i in range (0,len(rects)):
            area_lists.append(rects[i][2] * rects[i][3])
        temp_lists = Inference.radix_sort(area_lists)[-1:-5:-1]
        for i in range (0,len(rects)):
            if rects[i][2] * rects[i][3] in temp_lists:
                new_rects.append(rects[i])
                print(rects[i])
        return new_rects
    
    '''
    1 2
    3 4
    '''
    # 观察一下是否是邻近框变了
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
        return Point
        
    def compute_distance(Point1, Point2):
        distance = np.sqrt((Point1[0] - Point2[0]) ** 2 + (Point1[1] - Point2[1]) ** 2)
        return distance

    # 看起来没什么问题,前面偏差相对来说有点大，后面相对较小，补偿不好给（正对兑换站的情况  能识别角度 < 40度的样子
    def compute_pitch(distance_level_top, distance_level_borrom, distance_vertical_left, distance_vertical_right, self_angle):
        if distance_vertical_left > distance_level_top or distance_vertical_left > distance_level_borrom:                                
            pitch_angle = 0
        elif distance_vertical_right > distance_level_top or distance_vertical_right > distance_level_borrom:
            pitch_angle = 0
        else:
            pitch_radians0 = np.arcsin(distance_vertical_left / distance_level_top)
            pitch_radians1 = np.arcsin(distance_vertical_left / distance_level_borrom)
            pitch_radians2 = np.arcsin(distance_vertical_right / distance_level_top)
            pitch_radians3 = np.arcsin(distance_vertical_right / distance_level_borrom)
            # 还得减去自身角度
            pitch_angle = 90 - Inference.radians_to_angle((pitch_radians0 + pitch_radians1  + pitch_radians2 + pitch_radians3)/4) - self_angle
        return pitch_angle

    # 每5度加1 真实能够检测到的角度 < 40度 （正对兑换站的情况
    def compute_roll(top_left_point, top_right_point, bottom_left_point, bottom_right_point):
        # if bottom_right_point[1] == bottom_left_point[1] or top_right_point[1] == top_left_point[1]:
        #     roll_angle = 0
        # else:
        k0 = (bottom_right_point[1] - bottom_left_point[1]) / (bottom_right_point[0] - bottom_left_point[0])
        k1 = (top_right_point[1] - top_left_point[1]) / (top_right_point[0] - top_left_point[0])
        roll_radians_k0 = np.arctan(-k0)
        roll_radians_k1 = np.arctan(-k1)
        roll_angle = Inference.radians_to_angle((roll_radians_k0 + roll_radians_k1)/2)
        return roll_angle

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