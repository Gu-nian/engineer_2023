import cv2
import numpy as np

from inference_function.share_function import Share
class Station():
    
    deviation_x = 0
    direction = 2
    target_x = 640
    pitch_angle = 0
    roll_flag = 0
    roll_angle = 0
    def init_serial_data():
        Station.deviation_x = 0
        Station.direction = 2
        Station.pitch_angle = 0
        Station.roll_flag = 0
        Station.roll_angle = 0

    def set_serial_data(direction, deviation_x, pitch_angle, roll_flag, roll_angle):
        Station.direction = direction
        Station.deviation_x = deviation_x        
        Station.pitch_angle = pitch_angle
        Station.roll_flag = roll_flag
        Station.roll_angle = roll_angle

    # 13度是误差极限 还是位置极佳准确的时候
    def print_serial_data():
        print("deviation_x: ", Station.deviation_x)
        print("direction: ", Station.direction)
        print("pitch_angle: ", Station.pitch_angle)
        print("roll_flag: ", roll_flag)
        print("roll_angle: ", Station.roll_angle)
        print()

    # 传统视觉找灯条
    def find_light(frame):
        tohsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        toinRange = cv2.inRange(tohsv, (0, 0, 214), (109, 255, 255))
        # cv2.imshow('inrange', toinRange)
        contours, _ = cv2.findContours(toinRange, 0, cv2.CHAIN_APPROX_SIMPLE)  
        return contours

    # 选出最优兑换站
    def station_compare(frame, stations):
        img_size = frame.shape
        for i,station in enumerate(stations):
            tag, x_center, y_center, width, height = station
            width = float(width) * img_size[1]
            height = float(height) * img_size[0]
            area = int(width * height)
            area_temp = 0
            station_temp = 0
            if area > area_temp:
                area_temp = area
                station_temp = station
        tag, x_center, y_center, width, height = station_temp
        x_center, width = float(x_center) * img_size[1], float(width) * img_size[1]
        y_center, height = float(y_center) * img_size[0], float(height) * img_size[0]
        top_left = (int(x_center - width * 0.5), int(y_center - height * 0.5))
        top_right = (int(x_center + width * 0.5), int(y_center - height * 0.5))
        bottom_left = (int(x_center - width * 0.5), int(y_center + height * 0.5))
        bottom_right = (int(x_center + width * 0.5), int(y_center + height * 0.5))
        return x_center, y_center, top_left, top_right, bottom_left, bottom_right

    # 推理rects间包含关系
    def include_relationship(img_size, inference_rects, station_top_left, station_bottom_right):
        new_inference_rects = []
        for i,special_rect in enumerate(inference_rects):
            tag, x_center, y_center, width, height = special_rect
            x_center, width = float(x_center) * img_size[1], float(width) * img_size[1]
            y_center, height = float(y_center) * img_size[0], float(height) * img_size[0]
            center = [x_center, y_center]
            if center[0] > station_top_left[0] and center[1] > station_top_left[1] and center[0] < station_bottom_right[0] and center[1] < station_bottom_right[1]:
                new_inference_rects.append(special_rect)
        return new_inference_rects
        
    # 推理rects与cv_rects间包含关系
    def include_cv_relationship(img_size, inference_rects, cv_rects):
        new_cv_rects = []
        for i, inference_rect in enumerate(inference_rects):
            tag, x_center, y_center, width, height = inference_rect
            x_center, width = float(x_center) * img_size[1], float(width) * img_size[1]
            y_center, height = float(y_center) * img_size[0], float(height) * img_size[0]
            top_left = (int(x_center - width * 0.5), int(y_center - height * 0.5))            
            bottom_right = (int(x_center + width * 0.5), int(y_center + height * 0.5))
            for j, cv_rect in enumerate(cv_rects):      
                cv_top_left_x, cv_top_left_y, cv_width, cv_height = cv_rect
                if cv_width < 10 or cv_height < 10 :
                    continue
                cv_rect_center = cv_top_left_x + cv_width / 2, cv_top_left_y + cv_height / 2
                if cv_rect_center[0] > top_left[0] and  cv_rect_center[1] > top_left[1] and cv_rect_center[0] < bottom_right[0] and cv_rect_center[1] < bottom_right[1] :                    
                    new_cv_rects.append(cv_rect)
        return new_cv_rects

    # cv_rects间去重
    def cv_rects_compare(rects1, rects2):
        for _, rect2 in enumerate(rects2):
            if rect2 in rects1:
                rects1.remove((rect2))
        return rects1 + rects2

    # special_rect明确确定
    def pre_confirm_special_rect(img_size, special_rects, result_rects):
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

    # special_rect隐式确定
    def confirm_special_rect(img_size, special_rects, station_top_left, station_top_right, station_bottom_left ,station_bottom_right, result_rects):
        special_rect, single = Station.pre_confirm_special_rect(img_size, special_rects, result_rects)
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

            distance_top_left = Share.compute_distance(station_top_left, top_left)
            distance_top_right = Share.compute_distance(station_top_right, top_right)
            distance_bottom_left = Share.compute_distance(station_bottom_left, bottom_left)
            distance_bottom_right = Share.compute_distance(station_bottom_right, bottom_right)

            distance_list = [distance_top_left, distance_top_right, distance_bottom_left, distance_bottom_right]
            distance_list = Share.radix_sort(distance_list)
            if distance_top_right == distance_list[0]:
                result_rect.append(special_rect)
                single = 2
                break
            elif distance_top_left == distance_list[0]:
                temp_rect.append(special_rect)
                single = 1
        if len(result_rect):
            return result_rect[0], single
        elif len(temp_rect):
            return temp_rect[0], single
        else:
            return [], single
    
    # 出现上面两个为special_rects, 去掉一个确认为true的另一个加入为nomal_rects
    def two_special_rect_dealwith(special_rects, special_rect, nomal_rects, nomal_rects_tag = '2'):
        special_rects.remove(special_rect) 
        if len(nomal_rects) < 3:
            for i, rect in enumerate(special_rects):
                tag, x_center, y_center, width, height = rect
                rect = [nomal_rects_tag, x_center, y_center, width, height]
                nomal_rects.append(rect)
        return nomal_rects

    # 获取special_rects中的cv_rects
    def special_rects_gain_cv_rects(img_size, special_rect, cv_rects):
        cv_rects = Station.include_cv_relationship(img_size, [special_rect], cv_rects)   
        if len(cv_rects) == 1:
            return cv_rects[0]
        if len(cv_rects) > 1:
            area_cv_rects = []
            for i, cv_rect in enumerate(cv_rects):
                _, _, width, height = cv_rect
                area_cv_rects.append(width * height)
            max_area = Share.radix_sort(area_cv_rects)[-1]
            for i, cv_rect in enumerate(cv_rects):
                _, _, width, height = cv_rect
                if width * height == max_area:
                    return cv_rect

    # 根据到顶边的距离进行排列
    def distance_compare(nomal_cv_rects, special_point):
        distance_lists = [Share.compute_distance(special_point, (special_point[0], 0))]
        for i, rect in enumerate(nomal_cv_rects):
            analysis_distance = Share.compute_distance(rect, (rect[0], 0))
            distance_lists.append(int(analysis_distance))
        four_point_distance = Share.radix_sort(distance_lists)
        return four_point_distance

    # 分析出4个角点用于后面角度解算
    def analysis_other_point(nomal_cv_rects, top_right_cv_rect):        
        top_right_point = [top_right_cv_rect[0], top_right_cv_rect[1]]
        four_point_distance = Station.distance_compare(nomal_cv_rects, top_right_point)
        top_right_point_distance = Share.compute_distance(top_right_point, (top_right_point[0], 0))

        # 右上点最高
        if top_right_point_distance == four_point_distance[0]:
            if len(four_point_distance) == 5:
                four_point_distance, nomal_cv_rects = Station.edge_correct(four_point_distance, nomal_cv_rects, '1')
            for i, rect in enumerate(nomal_cv_rects):
                analysis_distance = Share.compute_distance(rect, (rect[0], 0))
                if analysis_distance == four_point_distance[1]:                    
                    top_left_rect = rect
                if analysis_distance == four_point_distance[2]:
                    bottom_right_rect = rect
                if analysis_distance == four_point_distance[3]:
                    bottom_left_rect = rect
            top_left_point, bottom_left_point, bottom_right_point = Station.other_compare(top_left_rect, bottom_right_rect, bottom_left_rect, '1')
        # 第二高   
        if top_right_point_distance == four_point_distance[1]:
            if len(four_point_distance) == 5:
                four_point_distance, nomal_cv_rects = Station.edge_correct(four_point_distance, nomal_cv_rects, '2')
            for i, rect in enumerate(nomal_cv_rects):             
                analysis_distance = Share.compute_distance(rect, (rect[0], 0))
                if analysis_distance == four_point_distance[0]:
                    top_left_rect = rect
                if analysis_distance == four_point_distance[3]:
                    bottom_right_rect = rect
                if analysis_distance == four_point_distance[2]:
                    bottom_left_rect = rect
            top_left_point, bottom_left_point, bottom_right_point = Station.other_compare(top_left_rect, bottom_right_rect, bottom_left_rect, '2')
        # 第三高（防止仰视原因出现的错误   
        if top_right_point_distance == four_point_distance[2]:
            if len(four_point_distance) == 5:
                four_point_distance, nomal_cv_rects = Station.edge_correct(four_point_distance, nomal_cv_rects, '3')
            for i, rect in enumerate(nomal_cv_rects):
                analysis_distance = Share.compute_distance(rect, (rect[0], 0))
                if analysis_distance == four_point_distance[0]:
                    top_left_point = [rect[0], rect[1]]
                if analysis_distance == four_point_distance[3]:
                    bottom_right_point = [rect[0] + rect[2], rect[1] + rect[3]]
                if analysis_distance == four_point_distance[1]:
                    bottom_left_point = [rect[0], rect[1] + rect[3]]
        top_right_point = [top_right_cv_rect[0] + top_right_cv_rect[2], top_right_cv_rect[1]]
        # print("top_right_point: ", top_right_point)       
        # print("top_left_point: ", top_left_point)
        # print("bottom_right_point: ", bottom_right_point)
        # print("bottom_left_point: ", bottom_left_point)        
        return top_right_point, top_left_point, bottom_left_point, bottom_right_point

    def edge_correct(four_point_distance, nomal_cv_rects, mode):
        thirds = 0
        fourth = 0
        if(mode == '1'):                
            for i, rect in enumerate(nomal_cv_rects):
                analysis_distance = Share.compute_distance(rect, (rect[0], 0))
                if analysis_distance == four_point_distance[2]:
                    thirds = rect
                    thirds_distance = analysis_distance
                if analysis_distance == four_point_distance[3]:
                    fourth = rect
                    fourth_distance = analysis_distance
                if thirds and fourth:
                    if thirds[0] < fourth[0]:
                        wait_del_rect, wait_del_distance = thirds, thirds_distance
                    else:
                        wait_del_rect, wait_del_distance = fourth, fourth_distance                                                         
            nomal_cv_rects.remove(wait_del_rect)
            four_point_distance.remove(wait_del_distance)
            return four_point_distance, nomal_cv_rects
        else:
            for i, rect in enumerate(nomal_cv_rects):
                analysis_distance = Share.compute_distance(rect, (rect[0], 0))
                if analysis_distance == four_point_distance[2]:
                    thirds = rect
                    thirds_distance = analysis_distance
                if analysis_distance == four_point_distance[3]:
                    fourth = rect
                    fourth_distance = analysis_distance
                if thirds and fourth:
                        if thirds[0] > fourth[0]:
                            wait_del_rect, wait_del_distance = thirds, thirds_distance
                        else:
                            wait_del_rect, wait_del_distance = fourth, fourth_distance  
            nomal_cv_rects.remove(wait_del_rect)
            four_point_distance.remove(wait_del_distance)
            return four_point_distance, nomal_cv_rects

    # 角点纠正
    def other_compare(top_left_rect, bottom_right_rect, bottom_left_rect, position):
        if position == '1':
            if top_left_rect[0] > bottom_right_rect[0]:
                temp_rect = top_left_rect
                top_left_rect = bottom_right_rect[0]
                bottom_right_rect = temp_rect
            if bottom_left_rect[0] > bottom_right_rect[0]:
                temp_rect = bottom_left_rect
                bottom_left_rect = bottom_right_rect
                bottom_right_rect = temp_rect
        if position == '2':
            if bottom_left_rect[0] > bottom_right_rect[0]:
                temp_rect = bottom_left_rect
                bottom_left_rect = bottom_right_rect
                bottom_right_rect = temp_rect
            
        top_left_point = [top_left_rect[0], top_left_rect[1]]
        bottom_left_point = [bottom_left_rect[0], bottom_left_rect[1] + bottom_left_rect[3]]
        bottom_right_point = [bottom_right_rect[0] + bottom_right_rect[2], bottom_right_rect[1] + bottom_right_rect[3]]
        return top_left_point, bottom_left_point, bottom_right_point

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
            pitch_angle = 90 - Station.radians_to_angle((pitch_radians0 + pitch_radians1  + pitch_radians2 + pitch_radians3)/4) - self_angle
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
            roll_angle = Station.radians_to_angle((roll_radians_k0 + roll_radians_k1)/2)
        return roll_angle

    # roll_angle补偿
    def roll_angle_compensate(roll_angle):
        roll_angle_compensate = int(roll_angle / 5)
        temp_roll_angle = roll_angle + roll_angle_compensate
        temp_roll_angle_compensate = int(temp_roll_angle/5)
        real_roll_angle = temp_roll_angle + temp_roll_angle_compensate - roll_angle_compensate
        return real_roll_angle

    # 弧度转角度
    def radians_to_angle(radians_value):
        PI = 3.14159265359
        angle = radians_value * 180 / PI
        return angle
    
