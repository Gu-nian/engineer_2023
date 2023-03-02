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
        mineral_arr = []
        rects = []  
        nomal_rects =  []
        nomal_rects_confs = []
        station = []
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

                    # mineral
                    if tag == 'mineral':                        
                        mineral_arr.append(int(x_center - Inference.TARGET_X)) 
                    # station
                    if tag == '1':      
                        station.append(det)     
                        station_confs.append(i)
                    # special_point
                    if tag == '0':
                        special_rects.append(det)
                        special_rects_confs.append(i)
                    # nomal_point
                    if tag == '2':
                        nomal_rects.append(det)
                        nomal_rects_confs.append(i)
                
                # mineral
                if  len(mineral_arr) > 0:
                    if abs(Inference.radix_sort(mineral_arr)[0]) < abs(Inference.radix_sort(mineral_arr)[len(mineral_arr)-1]):
                        Inference.DEVIATION_X = Inference.radix_sort(mineral_arr)[0]
                    else:
                        Inference.DEVIATION_X = Inference.radix_sort(mineral_arr)[len(mineral_arr)-1]

                    if mode == True:
                        cv2.putText(frame, "real_x = " + str(Inference.DEVIATION_X), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                    Inference.HIGH_EIGHT = (abs(Inference.DEVIATION_X) >> 8) & 0xff
                    Inference.LOW_EIGHT = abs(Inference.DEVIATION_X)  & 0xff
                    
                    if abs(Inference.DEVIATION_X ) < 24:
                        Inference.DEVIATION_X  = 0
                    if Inference.DEVIATION_X > 0:
                        Inference.DIRECTION = 1
                    Inference.draw_data(frame, img_size, mode)

                # 兑换站识别
                if len(station) > 0 and len(special_rects) > 0 and len(nomal_rects) > 0:
                    # 筛选面积最大的“兑换站”防止误识别
                    for i in range (0,len(station)):
                        temp = 0
                        if float(station[i][3]) * float(station[i][4]) > temp:
                            temp = float(station[i][3]) * float(station[i][4])                            
                            target = i
                    tag, x_center, y_center, width, height = station[target]
                    station_x_center, station_width = float(x_center) * img_size[1], float(width) * img_size[1]
                    station_y_center, station_height = float(y_center) * img_size[0], float(height) * img_size[0]
                    station_top_left = (int(station_x_center - station_width * 0.5), int(station_y_center - station_height * 0.5))
                    station_top_right = (int(station_x_center + station_width * 0.5), int(station_y_center - station_height * 0.5))
                    station_bottom_left = (int(station_x_center - station_width * 0.5), int(station_y_center + station_height * 0.5))
                    station_bottom_right = (int(station_x_center + station_width * 0.5), int(station_y_center + station_height * 0.5))

                    special_rects = Inference.include_relationship(station_top_left, station_bottom_right, special_rects, img_size)                    
                    nomal_rects = Inference.include_relationship(station_top_left, station_bottom_right, nomal_rects, img_size)

                    # 逐步筛选灯条
                    for i in range(0,len(contours)):
                        rect = cv2.boundingRect(contours[i])
                        top_left_x, top_left_y, width, height = rect
                        light_center = top_left_x + width / 2, top_left_y + height / 2
                        if light_center[0] > station_top_left[0] and  light_center[1] > station_top_left[1] and \
                            light_center[0] < station_bottom_right[0] and light_center[1] < station_bottom_right[1] :
                            rects.append(rect)

                    special_cv_rects = Inference.include_relationship_cv(special_rects, rects, img_size)
                    nomal_cv_rects = Inference.include_relationship_cv(nomal_rects, rects, img_size)
                    result_rects = Inference.rects_compare(special_cv_rects, nomal_cv_rects)
                    for i in range (0,len(result_rects)):
                        cv2.rectangle(frame,result_rects[i],(0,0,255),3)
                                    
                    # special_rects 与 nomal_rects 上边有概率重合
                    # 去除x最小那个
                    # 出现再解决                                       

                    # 可能情况缺少，具体情况具体分析                
                    special_rect, single = Inference.confirm_special_rect(special_rects, station_top_left, station_top_right, station_bottom_left, station_bottom_right, result_rects, img_size)
                    # if single == 0:
                        # continue
                    if single == 2:
                        top_right_cv_rect = Inference.special_rects_gain_cv_rects(special_rect, special_cv_rects, img_size)
                    top_right_point = 

                    try:                        
                        four_point_distance = Inference.distance_compare(nomal_rects, top_right_point)
                        top_left_point, bottom_left_point, bottom_right_point = Inference.analysis_other_point(nomal_rects, top_right_point, four_point_distance)
                        
                        # 均值处理角度 
                        distance_level_borrom = Inference.compute_distance(bottom_left_point, bottom_right_point)                        
                        distance_vertical_left = Inference.compute_distance(top_left_point, bottom_left_point)
                        distance_level_top = Inference.compute_distance(top_left_point, top_right_point)
                        distance_vertical_right = Inference.compute_distance(top_right_point, bottom_right_point)                        
                    except:
                        pass          
                    try:   
                        pitch_angle =Inference.compute_pitch(distance_level_top, distance_level_borrom, distance_vertical_left, distance_vertical_right, 31)  
                        roll_angle = Inference.compute_roll(top_left_point, top_right_point, bottom_left_point, bottom_right_point)
                        # 13度是误差极限 还是位置极佳准确的时候                    
                        print("pitch_angle: ", pitch_angle, "\n", "roll_angle: ", roll_angle)
                    except:               
                        pass         
                        # print("Analysis Angle Error")
    """
    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                    ***函数功能区***
    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    """
    # 传统视觉找灯条
    def find_light(frame):
        tohsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        toinRange = cv2.inRange(tohsv, (0, 0, 214), (109, 255, 255))
        cv2.imshow('inrange', toinRange)          
        contours, _ = cv2.findContours(toinRange, 0, cv2.CHAIN_APPROX_SIMPLE)  
        return contours

    # 面积筛选 排除roll旋转小灯条的干扰 保留4个
    def area_compare(rects):
        area_lists = []
        new_rects = []
        for i in range (0,len(rects)):
            area_lists.append(rects[i][2] * rects[i][3])
        temp_lists = Inference.radix_sort(area_lists)[-1:-9:-1]
        for i in range (0,len(rects)):
            if rects[i][2] * rects[i][3] in temp_lists:
                new_rects.append(rects[i])
        return new_rects
    def area_compare_max(rects):
        area_lists = []
        for i in range (0,len(rects)):
            area_lists.append(rects[i][2] * rects[i][3])
        max_area = Inference.radix_sort(area_lists)[-1]
        for i in range (0,len(rects)):
            if rects[i][2] * rects[i][3] == max_area:
                max_point = rects[i]
        return rects[i]

    # 去重
    def rects_compare(rects1, rects2):
        for _, i in enumerate(rects1):            
            i_top_left = (i[0], i[1])
            i_bottom_right = [(i[0] + i[2]), (i[1] + i[3])]
            temp = 0
            for num, j in enumerate(rects2):                
                j_center = [j[0] + j[2] / 2 , j[1] + j[3] /2]                
                if  j_center[0] < i_top_left[0] and j_center[1] < i_top_left[1] and j_center[0] > i_bottom_right[0] and j_center[1] > i_bottom_right[1]:
                    del rects2[num - temp]
                    temp += 1
        return rects1 + rects2

    def pre_confirm_special_rect(special_rects, result_rects, img_size):
        single = 0
        for i, special_rect in enumerate(special_rects):
            count = 0
            tag, x_center, y_center, width, height = special_rect
            x_center, width = float(x_center) * img_size[1], float(width) * img_size[1]
            y_center, height = float(y_center) * img_size[0], float(height) * img_size[0]
            top_left = (int(x_center - width * 0.5), int(y_center - height * 0.5))            
            bottom_right = (int(x_center + width * 0.5), int(y_center + height * 0.5))
            for j, result_rect in enumerate(result_rects):      
                result_rect_top_left_x, result_rect_top_left_y, result_rect_width, result_rect_height = result_rect                
                result_rect_center = result_rect_top_left_x + result_rect_width / 2, result_rect_top_left_y + result_rect_height / 2
                if result_rect_center[0] > top_left[0] and  result_rect_center[1] > top_left[1] and result_rect_center[0] < bottom_right[0] and result_rect_center[1] < bottom_right[1] :                    
                    count += 1
            if count == 3:
                single = 2
                return special_rect, single
        return [], single
            
    # 不能保证一定合理
    # 确认唯一
    def confirm_special_rect(special_rects, station_top_left, station_top_right, station_bottom_left ,station_bottom_right, result_rects, img_size):
        special_rect, single = Inference.pre_confirm_special_rect(special_rects, result_rects, img_size)
        if single == 2:
            return special_rect, single
        single = 0
        temp_rect = []
        result_rect = []
        for i, special_rect in enumerate(special_rects):
            tag, x_center, y_center, width, height = special_rect
            x_center, width = float(x_center) * img_size[1], float(width) * img_size[1]
            y_center, height = float(y_center) * img_size[0], float(height) * img_size[0]
            top_left = (int(x_center - width * 0.5), int(y_center - height * 0.5))
            top_right = (int(x_center + width * 0.5), int(y_center - height * 0.5))
            bottom_left = (int(x_center - width * 0.5), int(y_center + height * 0.5))
            bottom_right = (int(x_center + width * 0.5), int(y_center + height * 0.5))

            distance_top_left = Inference.compute_distance(station_top_left, top_left)
            distance_top_right = Inference.compute_distance(station_top_right, top_right)
            distance_bottom_left = Inference.compute_distance(station_bottom_left, bottom_left)
            distance_bottom_right = Inference.compute_distance(station_bottom_right, bottom_right)

            distance_list = [distance_top_left, distance_top_right, distance_bottom_left, distance_bottom_right]
            distance_list = Inference.radix_sort(distance_list)
            if distance_top_right == distance_list[0]:
                result_rect.append(special_rect)
                single = 2
                break
            elif distance_top_left == distance_list[0]:
                temp_rect.append(special_rect)
                single = 1
        if len(result_rect):
            print(result_rect[0])
            return result_rect[0], single
        elif len(temp_rect):
            print(temp_rect[0])
            return temp_rect[0], single
        else:
            return [], single

    def include_relationship(out_top_left, out_bottom_right, in_rects, img_size):
        new_rects = []
        for i,in_rect in enumerate(in_rects):
            tag, x_center, y_center, width, height = in_rect
            x_center, width = float(x_center) * img_size[1], float(width) * img_size[1]
            y_center, height = float(y_center) * img_size[0], float(height) * img_size[0]
            center = [x_center, y_center]
            if center[0] > out_top_left[0] and center[1] > out_top_left[1] and center[0] < out_bottom_right[0] and center[1] < out_bottom_right[1]:
                new_rects.append(in_rect)
        return new_rects

    def include_relationship_cv(out_rects, in_cv_rects, img_size):
        new_rects = []       
        for i, out_rect in enumerate(out_rects):
            tag, x_center, y_center, width, height = out_rect
            x_center, width = float(x_center) * img_size[1], float(width) * img_size[1]
            y_center, height = float(y_center) * img_size[0], float(height) * img_size[0]
            top_left = (int(x_center - width * 0.5), int(y_center - height * 0.5))            
            bottom_right = (int(x_center + width * 0.5), int(y_center + height * 0.5))
            for j, cv_rect in enumerate(in_cv_rects):      
                cv_top_left_x, cv_top_left_y, cv_width, cv_height = cv_rect
                if cv_width < 10 or cv_height < 10 :
                    continue
                light_center = cv_top_left_x + cv_width / 2, cv_top_left_y + cv_height / 2
                if light_center[0] > top_left[0] and  light_center[1] > top_left[1] and light_center[0] < bottom_right[0] and light_center[1] < bottom_right[1] :                    
                    new_rects.append(cv_rect)
        return new_rects

    # 分析出四灯条
    def distinguish_four_point(inference_rects, rects, frame, confs):
        img_size = frame.shape
        # special
        for i,inference_rect in enumerate(inference_rects):
            tag, x_center, y_center, width, height = inference_rect
            if tag == '0':                
                x_center, width = float(x_center) * img_size[1], float(width) * img_size[1]
                y_center, height = float(y_center) * img_size[0], float(height) * img_size[0]
                x_center, width = float(x_center) * img_size[1], float(width) * img_size[1]
                y_center, height = float(y_center) * img_size[0], float(height) * img_size[0]
                top_left = (int(x_center - width * 0.5), int(y_center - height * 0.5))
                top_right = (int(x_center + width * 0.5), int(y_center - height * 0.5))
                bottom_left = (int(x_center - width * 0.5), int(y_center + height * 0.5))
                bottom_right = (int(x_center + width * 0.5), int(y_center + height * 0.5))
                compare_rects = []
                result_rects = []
                result_center_points = []                
                for i in range (0,len(rects)):
                    rects_center = [rects[i][0] + rects[i][2] / 2, rects[i][1] + rects[i][3] / 2]
                    if rects_center[0] > top_left[0] and rects_center[1] > top_left[1] and rects_center[0] < bottom_right[0] and rects_center[1] < bottom_right[1]:
                        compare_rects.append(rects[i])
                        print(compare_rects)
                if len(compare_rects) > 1:
                    rect = Inference.area_compare_max(compare_rects)
                    center_point = [rect[0] + rect[2] / 2, rect[1] + rect[3] / 2]
                    result_rects.append(rect)
                    result_center_points.append(center_point)
                    cv2.rectangle(frame, rect, (0,0,0), 3)
                elif len(compare_rects) == 1:
                    rect = compare_rects[0]
                    center_point = [rect[0] + rect[2] / 2, rect[1] + rect[3] / 2]
                    result_rects.append(rect)
                    result_center_points.append(center_point)
                    cv2.rectangle(frame, rect, (0,0,0), 3)                
                return result_rects, result_center_points
        # nomal
            else:            
                result_rects = []
                result_center_points = []
                for i,rect in enumerate(inference_rects):
                    tag, x_center, y_center, width, height = rect
                    x_center, width = float(x_center) * img_size[1], float(width) * img_size[1]
                    y_center, height = float(y_center) * img_size[0], float(height) * img_size[0]
                    x_center, width = float(x_center) * img_size[1], float(width) * img_size[1]
                    y_center, height = float(y_center) * img_size[0], float(height) * img_size[0]
                    top_left = (int(x_center - width * 0.5), int(y_center - height * 0.5))
                    top_right = (int(x_center + width * 0.5), int(y_center - height * 0.5))
                    bottom_left = (int(x_center - width * 0.5), int(y_center + height * 0.5))
                    bottom_right = (int(x_center + width * 0.5), int(y_center + height * 0.5))
                    Inference.draw_inference(frame, top_left, top_right, bottom_right, tag, confs, i)
                    compare_rects = []
                    for j in range (0,len(rects)):
                        rects_center = [rects[j][0] + rects[j][2] / 2, rects[j][1] + rects[j][3] / 2]
                        if rects_center[0] > top_left[0] and rects_center[1] > top_left[1] and rects_center[0] < bottom_right[0] and rects_center[1] < bottom_right[1]:
                            compare_rects.append(rects[j])
                    if len(compare_rects) > 1:
                        rect = Inference.area_compare_max(compare_rects)
                        center_point = [rect[0] + rect[2] / 2, rect[1] + rect[3] / 2]
                        result_rects.append(rect)
                        result_center_points.append(center_point)
                        cv2.rectangle(frame, rect, (0,0,0), 3)
                    elif len(compare_rects) == 1:
                        rect = compare_rects[i][0]
                        center_point = [rect[0] + rect[2] / 2, rect[1] + rect[3] / 2]
                        result_rects.append(rect)
                        result_center_points.append(center_point)
                        cv2.rectangle(frame, rect, (0,0,0), 3)
                return result_rects, result_center_points
    
    # 深度学习rects转cv_rects
    def to_cv_rects(deep_learning_rects, img_size):
        cv_rects = []
        for i, deep_learning_rect in enumerate(deep_learning_rects):
            tag, x_center, y_center, width, height = deep_learning_rect
            x_center, width = float(x_center) * img_size[1], float(width) * img_size[1]
            y_center, height = float(y_center) * img_size[0], float(height) * img_size[0]
            top_left = (int(x_center - width * 0.5), int(y_center - height * 0.5))            
            temp_rect = [top_left[0], top_left[1], width, height]
            cv_rects.append(temp_rect)
        return cv_rects

    def special_rects_gain_cv_rects(special_rect, in_cv_rects, img_size):
        cv_rects = Inference.include_relationship_cv([special_rect], in_cv_rects, img_size)   
        if len(cv_rects) == 1:
            return cv_rects[0]     
        if len(cv_rects) > 1:
            area_cv_rects = []
            for i, cv_rect in enumerate(cv_rects):
                _, _, width, height = cv_rect
                area_cv_rects.append(width * height)
            max_area = Inference.radix_sort(area_cv_rects)[-1]
            for i, cv_rect in enumerate(cv_rects):
                _, _, width, height = cv_rect
                if width * height == max_area:
                    return cv_rect

    def nomal_rects_gain_cv_rects(nomal_rects, in_cv_rects, img_size):
        pass
    
    # 根据到顶边的距离进行排列
    def distance_compare(rects, special_point):
        distance_lists = [Inference.compute_distance(special_point, (special_point[0], 0))]
        for i in range(0,len(rects)):
            analysis_distance = Inference.compute_distance(rects[i], (rects[i][0], 0))
            distance_lists.append(int(analysis_distance))
        four_point_distance = Inference.radix_sort(distance_lists)
        return four_point_distance

    # 确定其他3个点的坐标
    def analysis_other_point(rects, top_right_point, four_point_distance):        
        # 右上点最高
        if Inference.compute_distance(top_right_point, (top_right_point[0], 0)) == four_point_distance[0]:
            print('1')
            for i in range(0, len(rects)):
                analysis_distance = Inference.compute_distance(rects[i], (rects[i][0], 0))
                if analysis_distance == four_point_distance[1]:
                    top_left_point = [rects[i][0] , rects[i][1]]
                if analysis_distance == four_point_distance[2]:
                    bottom_right_point = [rects[i][0] + rects[i][2], rects[i][1] + rects[i][3]]
                if analysis_distance == four_point_distance[3]:
                    bottom_left_point = [rects[i][0], rects[i][1] + rects[i][3]]
        # 第二高
        if Inference.compute_distance(top_right_point, (top_right_point[0], 0)) == four_point_distance[1]:
            print('2')
            for i in range(0, len(rects)):
                analysis_distance = Inference.compute_distance(rects[i], (rects[i][0], 0))
                if analysis_distance == four_point_distance[0]:
                    top_left_point = [rects[i][0], rects[i][1]]
                if analysis_distance == four_point_distance[3]:
                    bottom_right_point = [rects[i][0] + rects[i][2], rects[i][1] + rects[i][3]]
                if analysis_distance == four_point_distance[2]:
                    bottom_left_point = [rects[i][0], rects[i][1] + rects[i][3]]
        # 第三高（防止仰视原因出现的错误
        if Inference.compute_distance(top_right_point, (top_right_point[0], 0)) == four_point_distance[2]:
            print('3')
            for i in range(0, len(rects)):
                analysis_distance = Inference.compute_distance(rects[i], (rects[i][0], 0))
                if analysis_distance == four_point_distance[0]:
                    top_left_point = [rects[i][0], rects[i][1]]
                if analysis_distance == four_point_distance[3]:
                    bottom_right_point = [rects[i][0] + rects[i][2], rects[i][1] + rects[i][3]]
                if analysis_distance == four_point_distance[1]:
                    bottom_left_point = [rects[i][0], rects[i][1] + rects[i][3]]
        return top_left_point, bottom_left_point, bottom_right_point
    
    # 计算距离
    def compute_distance(Point1, Point2):
        distance = np.sqrt((Point1[0] - Point2[0]) ** 2 + (Point1[1] - Point2[1]) ** 2)
        return int(distance)
   
    # pitch值计算
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

    # roll计算
    def compute_roll(top_left_point, top_right_point, bottom_left_point, bottom_right_point):
        if bottom_right_point[1] == bottom_left_point[1] or top_right_point[1] == top_left_point[1]:
            roll_angle = 0
        else:
            k0 = (bottom_right_point[1] - bottom_left_point[1]) / (bottom_right_point[0] - bottom_left_point[0])
            k1 = (top_right_point[1] - top_left_point[1]) / (top_right_point[0] - top_left_point[0])
            roll_radians_k0 = np.arctan(-k0)
            roll_radians_k1 = np.arctan(-k1)
            roll_angle = Inference.radians_to_angle((roll_radians_k0 + roll_radians_k1)/2)
        return roll_angle

    # 弧度转角度
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