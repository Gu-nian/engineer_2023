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
    def draw_inference(frame, img_size, inference_rects, mode = 1):
        if mode == True:
            for i,inference_rect in enumerate(inference_rects):
                tag, x_center, y_center, width, height = inference_rect
                x_center, width = float(x_center) * img_size[1], float(width) * img_size[1]
                y_center, height = float(y_center) * img_size[0], float(height) * img_size[0]
                top_left = (int(x_center - width * 0.5), int(y_center - height * 0.5))
                top_right = (int(x_center + width * 0.5), int(y_center - height * 0.5))
                bottom_left = (int(x_center - width * 0.5), int(y_center + height * 0.5))
                bottom_right = (int(x_center + width * 0.5), int(y_center + height * 0.5))
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 255), 3, 8)
                cv2.putText(frame, tag, top_left, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 4)
    
     # 将数据显示出来
    # def draw_data(frame, img_size, mode = 1):
    #     if mode == True:
    #         cv2.putText(frame, "judge_x = " + str(Inference.DEVIATION_X), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    #         cv2.line(frame, (Inference.TARGET_X, 0), (Inference.TARGET_X, int(img_size[0])), (255, 0, 255), 3)
    #         cv2.putText(frame, 'direction: ' + str(Inference.DIRECTION), (0, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    #         cv2.putText(frame, 'high_eight: ' + str(Inference.HIGH_EIGHT), (0, 210), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    #         cv2.putText(frame, 'low_eight: ' + str(Inference.LOW_EIGHT), (0, 260), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)