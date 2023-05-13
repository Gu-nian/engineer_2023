from pickle import FALSE
import time

import cv2
import numpy as np

from video_function import mvsdk

class Video_capture:
    COLS = 1280
    ROWS = 800
    # ExposureTime = 12 * 1000
    ExposureTime = 20 * 1000
    IS_SAVE_VIDEO = 0
    # 判断相机是否掉线
    CAMERA_OPEN = 0
    # 相机初始化配置
    def __init__(self,is_save_video = 0):

        Video_capture.IS_SAVE_VIDEO = is_save_video

        DevList = mvsdk.CameraEnumerateDevice()
        try:
            for i, DevInfo in enumerate(DevList):
                ("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
            i = 0 
            DevInfo = DevList[i]
            print(DevInfo)
            # 打开相机
            self.hCamera = 0
            try:
                self.hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
            except mvsdk.CameraException as e:
                print("CameraInit Failed({}): {}".format(e.error_code, e.message) )
            
            # 录制视频
            if Video_capture.IS_SAVE_VIDEO :
                try:
                    print(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
                    time_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                    # print(time_name)
                    sfourcc = cv2.VideoWriter_fourcc(*'XVID')#视频存储的格式
                    #视频的宽高
                    self.out = cv2.VideoWriter('./video/' + time_name + '.avi', sfourcc, 30, (Video_capture.COLS,Video_capture.ROWS))#视频存储
                except:
                    print("To Save Video Error")
                    
            # 获取相机特性描述
            cap = mvsdk.CameraGetCapability(self.hCamera)

            # 判断是黑白相机还是彩色相机
            monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

            # 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
            if monoCamera:
                mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
            else:
                mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

            # 相机模式切换成连续采集
            mvsdk.CameraSetTriggerMode(self.hCamera, 0)

            pImageResolution = mvsdk.CameraGetImageResolution(self.hCamera)
            pImageResolution.iIndex       = 0xFF
            pImageResolution.iWidthFOV    = Video_capture.COLS
            pImageResolution.iHeightFOV   = Video_capture.ROWS
            pImageResolution.iWidth       = Video_capture.COLS
            pImageResolution.iHeight      = Video_capture.ROWS
            pImageResolution.iHOffsetFOV  = int((1280 - Video_capture.COLS) * 0.5)
            pImageResolution.iVOffsetFOV  = int((1024 - Video_capture.ROWS) * 0.5) 

            mvsdk.CameraSetImageResolution(self.hCamera, pImageResolution)

            # 手动曝光，曝光时间30ms
            mvsdk.CameraSetAeState(self.hCamera, 0)
            mvsdk.CameraSetExposureTime(self.hCamera, Video_capture.ExposureTime )
            # 颜色补偿
            mvsdk.CameraSetGain(self.hCamera, 130, 119, 100)                         

            # 让SDK内部取图线程开始工作
            mvsdk.CameraPlay(self.hCamera)

            # 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
            FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)

            # 分配RGB buffer，用来存放ISP输出的图像
            # 备注：从相机传输到PC端的是RAW数据，在PC端通过软件ISP转为RGB数据（如果是黑白相机就不需要转换格式，但是ISP还有其它处理，所以也需要分配这个buffer）
            self.pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)
            Video_capture.CAMERA_OPEN = 1
        except:
            print('Not Find Camera')
    

    # 只开启摄像头
    def only_capture(self):
        while (cv2.waitKey(1) & 0xFF) != ord('q'):
            # 从相机取一帧图片
            try:
                pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.hCamera, 200)
                mvsdk.CameraImageProcess(self.hCamera, pRawData, self.pFrameBuffer, FrameHead)
                mvsdk.CameraReleaseImageBuffer(self.hCamera, pRawData)
                
                # 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
                # 把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
                frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(self.pFrameBuffer)
                frame = np.frombuffer(frame_data, dtype=np.uint8)
                frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3) )

                cv2.imshow("frame",frame)

                if Video_capture.IS_SAVE_VIDEO:
                    try:
                        self.out.write(frame)
                    except:
                        print("Write Frame Error")
                
            except mvsdk.CameraException as e:
                if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                    print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message) )  

        if Video_capture.IS_SAVE_VIDEO:
            try:
                self.out.release()
            except:
                print("Release Frame Error")

        mvsdk.CameraUnInit(self.hCamera)
        mvsdk.CameraAlignFree(self.pFrameBuffer)

    def empty(a):
        h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
        h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
        s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
        s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
        v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
        v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
        print(h_min, h_max, s_min, s_max, v_min, v_max)
        return h_min, h_max, s_min, s_max, v_min, v_max

    def only_capture_hsv(self):
        cv2.namedWindow("TrackBars")
        cv2.resizeWindow("TrackBars",640,240)
        cv2.createTrackbar("Hue Min","TrackBars",0,179,Video_capture.empty)
        cv2.createTrackbar("Hue Max","TrackBars",19,179,Video_capture.empty)
        cv2.createTrackbar("Sat Min","TrackBars",110,255,Video_capture.empty)
        cv2.createTrackbar("Sat Max","TrackBars",240,255,Video_capture.empty)
        cv2.createTrackbar("Val Min","TrackBars",153,255,Video_capture.empty)
        cv2.createTrackbar("Val Max","TrackBars",255,255,Video_capture.empty)
        while (cv2.waitKey(1) & 0xFF) != ord('q'):
            # 从相机取一帧图片
            try:
                pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.hCamera, 200)
                mvsdk.CameraImageProcess(self.hCamera, pRawData, self.pFrameBuffer, FrameHead)
                mvsdk.CameraReleaseImageBuffer(self.hCamera, pRawData)
                
                # 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
                # 把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
                frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(self.pFrameBuffer)
                frame = np.frombuffer(frame_data, dtype=np.uint8)
                frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3) )
                imgHSV = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
                h_min,h_max,s_min,s_max,v_min,v_max = Video_capture.empty(0)
                lower = np.array([h_min,s_min,v_min])
                upper = np.array([h_max,s_max,v_max])
                mask = cv2.inRange(imgHSV,lower,upper)
                imgResult = cv2.bitwise_and(frame,frame,mask=mask)
                cv2.imshow("Mask", mask)
                cv2.imshow("Result", imgResult)
                cv2.imshow("frame",frame)

                if Video_capture.IS_SAVE_VIDEO:
                    try:
                        self.out.write(frame)
                    except:
                        print("Write Frame Error")
                
            except mvsdk.CameraException as e:
                if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                    print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message) )  

        if Video_capture.IS_SAVE_VIDEO:
            try:
                self.out.release()
            except:
                print("Release Frame Error")

        mvsdk.CameraUnInit(self.hCamera)
        mvsdk.CameraAlignFree(self.pFrameBuffer)

if __name__ == "__main__" :

    video = Video_capture(0)
    video.only_capture_hsv()

