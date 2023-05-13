from typing import List

import cv2
import numpy as np
class Share():
    # 快速排序 小到大
    def radix_sort(arr:List[int]):
        n = len(str(max(arr)))
        for k in range(n):
            bucket_list=[[] for i in range(10)]
            for i in arr:
                bucket_list[i//(10**k)%10].append(i)
            arr=[j for i in bucket_list for j in i]
        return arr

    # 计算距离
    def compute_distance(Point1, Point2):
        distance = np.sqrt((Point1[0] - Point2[0]) ** 2 + (Point1[1] - Point2[1]) ** 2)
        return int(distance)

    # 绘制推理框
    def draw_inference(frame, img_size, inference_rects, mode = 1, color = (0, 255, 255)):
        if mode == True:
            for i,inference_rect in enumerate(inference_rects):
                tag, x_center, y_center, width, height = inference_rect
                x_center, width = float(x_center) * img_size[1], float(width) * img_size[1]
                y_center, height = float(y_center) * img_size[0], float(height) * img_size[0]
                top_left = (int(x_center - width * 0.5), int(y_center - height * 0.5))
                top_right = (int(x_center + width * 0.5), int(y_center - height * 0.5))
                bottom_left = (int(x_center - width * 0.5), int(y_center + height * 0.5))
                bottom_right = (int(x_center + width * 0.5), int(y_center + height * 0.5))
                cv2.rectangle(frame, top_left, bottom_right, color, 3, 8)
                cv2.putText(frame, tag, top_left, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 4)

     # 将数据显示出来
    def draw_station_data(frame, img_size, direction, deviation_x, pitch_angle = 0, roll_flag = 0, roll_angle = 0, mode = 1):
        if mode == True:
            cv2.putText(frame, "deviation_x = " + str(deviation_x), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.line(frame, (640, 0), (640, int(img_size[0])), (255, 0, 255), 3)
            cv2.putText(frame, 'direction: ' + str(direction), (0, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(frame, 'pitch_angle: ' + str(pitch_angle), (0, 210), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(frame, 'roll_flag: ' + str(roll_flag), (0, 260), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(frame, 'roll_angle: ' + str(roll_angle), (0, 310), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    
    def draw_mineral_data(frame, img_size, direction, deviation_x,mode = 1):
        if mode == True:
            cv2.putText(frame, "deviation_x = " + str(deviation_x), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.line(frame, (640, 0), (640, int(img_size[0])), (255, 0, 255), 3)
            cv2.putText(frame, 'direction: ' + str(direction), (0, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)         