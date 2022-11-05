import argparse
import threading
import numpy as np

import cv2

import video_capture  
import mvsdk
from to_inference import Inference
from utils.torch_utils import time_sync
from use_serial import Interactive_serial

# 传入模型位置
def parse_opt():
    parser = argparse.ArgumentParser()
    # 自启动 default 要改成绝对路径
    parser.add_argument('--weights', nargs='+', type=str, default='/home/oyc/workspace/python_thread/best.pt', help='model path(s)')
    opt = parser.parse_args()
    return opt

def run( Inference, is_save = 0, mode = 1):
    
    frame = cv2.imread("/home/oyc/图片/1.png")
    # frame = cv2.resize(frame,(1280,800))
    Inference.to_inference(frame, Inference.device, Inference.model, Inference.imgsz, Inference.stride, mode=mode)
    cv2.imshow("frame",frame)        
    if (cv2.waitKey(0) & 0xFF) != ord('q'):
        cv2.destroyAllWindows()
if __name__ == "__main__":
    opt = parse_opt()
    Inf = Inference(**vars(opt))    
    run(Inf)

    
