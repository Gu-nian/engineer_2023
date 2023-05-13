from pickle import FALSE
from typing import List

import cv2
import numpy as np
import torch

import video_function.video_capture
from inference_function.station_function import Station
from inference_function.share_function import Share
from inference_function.mineral_function import Mineral
from video_function.video_capture import Video_capture
from models.common import DetectMultiBackend
from utils.general import check_img_size,non_max_suppression,scale_coords, xyxy2xywh
from utils.torch_utils import select_device

class Inference(object):
    # 判断模式
    FLAG = 1
    def __init__(self,weights):
        # 加载模型
        self.device = select_device('cpu')
        self.model = DetectMultiBackend(weights, device=self.device)
        self.stride = self.model.stride 
        self.imgsz = check_img_size((320,320),s=self.stride)
        self.model.model.float()

    def load_frame(self,frame):
        frame = frame

    
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

    def to_mineral_inference(self,frame, device, model, imgsz, stride, mode = 1, conf_thres=0.45, iou_thres=0.45):
        img_size = frame.shape
        img0 = frame 
        img = Inference.letterbox(img0,imgsz,stride=stride)[0]
        img = img.transpose((2,0,1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.
        
        if len(img.shape) == 3:
            img = img[None]

        pred = model(img)
        pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)        
        aims = []
        confs = []
        mineral_arr = []
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
                    if float(conf) > 0.8:
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
                    Share.draw_inference(frame, img_size, [det], mode)
                    cv2.putText(frame,str(float(round(confs[i], 2))), top_right, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

                    # mineral
                    if tag == '0' :
                        mineral_arr.append(int(x_center - Mineral.target_x)) 
                if  len(mineral_arr) > 0:
                    if abs(Share.radix_sort(mineral_arr)[0]) < abs(Share.radix_sort(mineral_arr)[-1]):
                        mineral_deviation_x = Share.radix_sort(mineral_arr)[0]
                    else:
                        mineral_deviation_x = Share.radix_sort(mineral_arr)[-1]

                    if mode == True:
                        cv2.putText(frame, "real_x = " + str(mineral_deviation_x), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

                    mineral_direction = 2       
                    if abs(mineral_deviation_x) < 10:
                        mineral_deviation_x  = 0
                    elif mineral_deviation_x < 0:
                        mineral_direction = 0
                    else:
                        mineral_direction = 1

                    if mode == True:
                        Share.draw_mineral_data(frame, img_size, mineral_direction, mineral_deviation_x)
                    Mineral.set_serial_data(abs(mineral_deviation_x), mineral_direction)
                    # Mineral.print_serial_data()
                else:
                    Mineral.init_serial_data()
# -------------------------------------------------------------------------------------------------------------------------------------------------------
    # 进行推理 绘制图像 结算出最优 发送数据
    def to_station_inference(self,frame, device, model, imgsz, stride, mode = 1, conf_thres=0.45, iou_thres=0.45):
        img_size = frame.shape
        img0 = frame 
        img = Inference.letterbox(img0,imgsz,stride=stride)[0]
        img = img.transpose((2,0,1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.
        
        contours = Station.find_light(frame)

        if len(img.shape) == 3:
            img = img[None]

        pred = model(img)
        pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)        
        aims = []
        confs = []
        rects = []  
        nomal_rects =  []
        nomal_rects_confs = []
        stations = []
        station_confs = []
        special_rects = []
        special_rects_confs = []
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
                    if float(conf) > 0.70:
                        aims.append(aim)
                        confs.append(float(conf))

            if len(aims):
                for i,det in enumerate(aims):
                    tag, x_center, y_center, width, height = det
                    # x_center, width = float(x_center) * img_size[1], float(width) * img_size[1]
                    # y_center, height = float(y_center) * img_size[0], float(height) * img_size[0]
                    # top_left = (int(x_center - width * 0.5), int(y_center - height * 0.5))
                    # top_right = (int(x_center + width * 0.5), int(y_center - height * 0.5))
                    # bottom_left = (int(x_center - width * 0.5), int(y_center + height * 0.5))
                    # bottom_right = (int(x_center + width * 0.5), int(y_center + height * 0.5))
                    # Share.draw_inference(frame, img_size, [det], mode)
                    # cv2.putText(frame,str(float(round(confs[i], 2))), top_right, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

                    # station
                    if tag == '1':      
                        stations.append(det)     
                        station_confs.append(confs[i])
                    # special_rects
                    if tag == '0':
                        special_rects.append(det)
                        special_rects_confs.append(confs[i])    
                    # nomal_rects
                    if tag == '2':
                        nomal_rects.append(det)
                        nomal_rects_confs.append(confs[i])
                
                # 等个数据接收后处理的东西
                if len(stations) > 0  and len(nomal_rects) > 0 :
                    station_x_center, station_y_center, station_top_left, station_top_right, station_bottom_left, station_bottom_right = Station.station_compare(frame, stations)
                    station_deviation_x  = station_x_center - Station.target_x

                    
                    special_rects = Station.include_relationship(img_size, special_rects, station_top_left, station_bottom_right )                    
                    nomal_rects = Station.include_relationship(img_size, nomal_rects, station_top_left, station_bottom_right)
                    
                    # 逐步筛选灯条
                    for i in range(0,len(contours)):
                        rect = cv2.boundingRect(contours[i])
                        top_left_x, top_left_y, width, height = rect
                        light_center = top_left_x + width / 2, top_left_y + height / 2
                        if light_center[0] > station_top_left[0] and  light_center[1] > station_top_left[1] and \
                            light_center[0] < station_bottom_right[0] and light_center[1] < station_bottom_right[1] :
                            rects.append(rect)
                    
                    special_rects, nomal_rects = Station.tackle_inference_rects(img_size, special_rects, nomal_rects, rects) 
                    special_cv_rects = Station.include_cv_relationship(img_size, special_rects, rects)
                    nomal_cv_rects = Station.include_cv_relationship(img_size, nomal_rects, rects)
                    result_rects = Station.cv_rects_compare(special_cv_rects, nomal_cv_rects)
                    # for i in range (0,len(result_rects)):
                    #     cv2.rectangle(frame,result_rects[i],(0,0,255),3)
                    
                    # special_rect, signal = Station.confirm_special_rect(img_size, special_rects, station_top_left, station_top_right, station_bottom_left, station_bottom_right, result_rects)
                    special_rect, signal = Station.pre_confirm_special_rect(img_size, special_rects, result_rects)
                    
                    if len(special_rects) > 1:
                        special_rects, nomal_rects = Station.two_special_rect_taskle(special_rects, special_rect, nomal_rects, '2')
                        nomal_cv_rects = Station.include_cv_relationship(img_size, nomal_rects, result_rects)
                    
                    if signal == 2:                        
                        top_right_cv_rect = Station.special_rects_gain_cv_rects(img_size, special_rect, special_cv_rects)
                    else:
                        continue
                    
                   
                    try:
                        Share.draw_inference(frame, img_size, [special_rect], mode, (255, 0, 255))
                        Share.draw_inference(frame, img_size, nomal_rects, mode)
                        top_right_point, top_left_point, bottom_left_point, bottom_right_point = Station.analysis_other_point(nomal_cv_rects, top_right_cv_rect)

                        distance_level_borrom = Share.compute_distance(bottom_left_point, bottom_right_point)                        
                        distance_vertical_left = Share.compute_distance(top_left_point, bottom_left_point)
                        distance_level_top = Share.compute_distance(top_left_point, top_right_point)                    
                        distance_vertical_right = Share.compute_distance(top_right_point, bottom_right_point)                                                                    
                    
                        pitch_angle = Station.compute_pitch(distance_level_top, distance_level_borrom, distance_vertical_left, distance_vertical_right, 23)  
                        pitch_angle = pitch_angle if pitch_angle > 0 else 0
                        roll_angle = Station.compute_roll(top_left_point, top_right_point, bottom_left_point, bottom_right_point)                        
                        roll_angle = Station.roll_angle_compensate(roll_angle)

                        station_direction = 2 
                        if abs(station_deviation_x) < 10:
                            station_deviation_x  = 0
                        elif station_deviation_x  > 0:
                            station_direction = 1
                        else:
                            station_direction = 0

                        roll_flag = 1 if roll_angle > 0 else 0
                        Station.set_serial_data(station_direction, round(abs(station_deviation_x)), round(pitch_angle), roll_flag, round(abs(roll_angle)))
                        # Station.print_serial_data()
                    except:
                        Station.init_serial_data()
                        print("Nomal_cv_rects Nums Error", " or ", "Analysis Angle Error", " or ", "Serial Error")                        
                else:
                    Station.init_serial_data()                
                    
            
    
   