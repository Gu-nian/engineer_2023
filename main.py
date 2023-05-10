import argparse
import threading
import cv2
import numpy as np
import time

from video_function import video_capture, mvsdk
from inference_function.to_inference import Inference
from inference_function.share_function import Share
from utils.torch_utils import time_sync
from serial_function.serial_function import Interactive_serial


# 传入模型位置
def parse_opt():
    parser = argparse.ArgumentParser()
    # 自启动 default 要改成绝对路径
    parser.add_argument('--weights_station', nargs='+', type=str, default='/home/nuc11-rm2/workspace/helpful/RM2023_model/station.pt', help='model path(s)')
    parser.add_argument('--weights_mineral', nargs='+', type=str, default='./inference_models/mineral.pt', help='model path(s)')
    opt = parser.parse_args()
    return opt

def run(Video, station, mineral, is_save = 0):
    while (cv2.waitKey(1) & 0xFF) != ord('q'):
        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(Video.hCamera, 200)
            mvsdk.CameraImageProcess(Video.hCamera, pRawData, Video.pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(Video.hCamera, pRawData)
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(Video.pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3) )
            station.to_station_inference(frame, station.device, station.model, station.imgsz, station.stride, mode=mode)
            # mineral.to_mineral_inference(frame, mineral.device, mineral.model, mineral.imgsz, mineral.stride, mode=mode)
            

            if Video.IS_SAVE_VIDEO:
                try:                    
                    Video.out.write(frame)
                except:
                    print("Write Frame Error")
            cv2.imshow('frame', frame)
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
    mvsdk.CameraUnInit(Video.hCamera)
    mvsdk.CameraAlignFree(Video.pFrameBuffer)

if __name__ == "__main__":

    opt = parse_opt()
    is_save = 0
    mode = 1
    while video_capture.Video_capture.CAMERA_OPEN == 0:
        Video = video_capture.Video_capture(is_save)

    station = Inference('/home/nuc11-rm2/workspace/helpful/RM2023_model/station.pt')
    # station = Inference('./inference_models/station.pt')
    mineral = Inference('./inference_models/best.pt')
    
    # Inference_serial = Interactive_serial()
    # Inference_serial.send_test_data()
    # Inference_serial.receive_data()
    # thread1 = threading.Thread(target=(Inference_serial.send_mineral_data), daemon = True)
    # thread2 = threading.Thread(target=(Inference_serial.send_station_data), daemon = True)
    # thread1.start()
    # thread2.start()

    run(Video, station, mineral, is_save)