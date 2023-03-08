import serial
import time
         
from inference_function.mineral_function import Mineral
from inference_function.station_function import Station
class Interactive_serial(object):
    def __init__(self):
        self.ser = serial.Serial()
        self.ser.port = "/dev/ttyUSB0"
        self.ser.baudrate = 921600
        self.ser.bytesize = 8
        self.ser.parity = 'N'
        self.ser.stopbits = 1
        try:
            self.ser.open()
        except:
            self.ser.close()
            print("Serial Open Error")
    # 串口掉线重连
    def serial_connection(self):
        self.ser.port = "/dev/ttyUSB0"
        self.ser.baudrate = 921600
        self.ser.bytesize = 8
        self.ser.parity = 'N'
        self.ser.stopbits = 1
        try:
            self.ser.open()
            print('Reconnection Success')
        except:
            self.ser.close()
            print("Serial Reconnection Error")

    # 串口发送移动信息 2停 0左 1右
    def send_mineral_data(self):
        while True:
            time.sleep(0.0005)
            try:
                if Mineral.deviation_x == 0:
                    self.ser.write(('S' + str(2) + str(0) + str(0) + str(0) + 'E').encode("utf-8"))
                elif Mineral.deviation_x / 100 >= 1:
                    self.ser.write(('S' + str(Mineral.direction) + str(Mineral.deviation_x) + 'E').encode("uft-8"))
                elif Mineral.deviation_x / 10 >= 1:
                    self.ser.write(('S' + str(Mineral.direction) + str(0) + str(Mineral.deviation_x) + 'E').encode("uft-8"))
                elif Mineral.deviation_x / 1 >= 1:
                    self.ser.write(('S' + str(Mineral.direction) + str(0) + str(0) + str(Mineral.deviation_x) + 'E').encode("uft-8"))
                else:
                    self.ser.write(('S' + str(2) + str(0) + str(0) + str(0) + 'E').encode("utf-8"))
            except:
                self.ser.close()
                print('Serial Send Mineral Data Error')
                Interactive_serial.serial_connection(self)
    
    def send_station_data(self):
        while True:
            time.sleep(0.0005)
            try:
                if Station.deviation_x == 0:
                    self.ser.write(('S' + str(2) + str(0) + str(0) + str(0) + 'E').encode("utf-8"))
            except:
                self.ser.close()
                print('Serial Send Station Data Error')
                Interactive_serial.serial_connection(self)
    # 串口接收数据
    # def receive_data(self):
    #     while True:
    #         time.sleep(0.05)
    #         try:
    #             data = self.ser.read(3)
    #             if data == b'\x03\x03\x03' or data == b'\x01\x01\x01':
    #                 Mineral.TARGET_X = 480  #空接 不抬升500 抬升480 
    #                 Mineral.FLAG = 1
    #                 # print(data)
    #             if data == b'\x02\x02\x02':
    #                 Mineral.TARGET_X = 415  #资源岛
    #                 Mineral.FLAG = 0
    #                 # print(data)
    #             print(data)
    #         except:                
    #             self.ser.close()
    #             print('Serial Send Data Error')
    #             Mineral.serial_connection(self)
                