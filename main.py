import argparse
import threading
import cv2
import numpy as np

import video_capture  
import mvsdk
from to_inference import Inference
from utils.torch_utils import time_sync
from use_serial import Interactive_serial

# 传入模型位置
def parse_opt():
    parser = argparse.ArgumentParser()
    # 自启动 default 要改成绝对路径
    parser.add_argument('--weights', nargs='+', type=str, default='./station.pt', help='model path(s)')
    opt = parser.parse_args()
    return opt

def run(Video, Inference, is_save = 0, mode = 1):
    
    while (cv2.waitKey(1) & 0xFF) != ord('q'):
        try:
            t2 = time_sync()
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(Video.hCamera, 200)
            mvsdk.CameraImageProcess(Video.hCamera, pRawData, Video.pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(Video.hCamera, pRawData)
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(Video.pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3) )
            # frame1 = video_capture.Video_capture.img_init(frame)
            Inference.to_inference(frame, Inference.device, Inference.model, Inference.imgsz, Inference.stride, mode=mode)
            
            t3 = time_sync()

            if Video.IS_SAVE_VIDEO:
                try:                    
                    Video.out.write(frame)
                except:
                    print("Write Frame Error")
            cv2.imshow("frame",frame)
            # print("Inference == " + str(1/(t3 - t2)))
        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message) )
            cv2.destroyAllWindows()
            video_capture.Video_capture.__init__(is_save)
    if Video.IS_SAVE_VIDEO:
        try:
            Video.out.release()
        except:
            print("Release Frame Error")
    cv2.destroyAllWindows()
    # 关闭相机
    mvsdk.CameraUnInit(Video.hCamera)
    # 释放帧缓存
    mvsdk.CameraAlignFree(Video.pFrameBuffer)
if __name__ == "__main__":

    opt = parse_opt()
    is_save = 0
    mode = 1
    while video_capture.Video_capture.CAMERA_OPEN == 0:
        Video = video_capture.Video_capture(is_save)

    Inference = Inference(**vars(opt))
    
    Inference_serial = Interactive_serial()
    thread1 = threading.Thread(target=(Inference_serial.send_mineral_data))
    thread2 = threading.Thread(target=(Inference_serial.send_station_data))
    thread1.start()
    thread2.start()
    
    run(Video,Inference,is_save,mode)
