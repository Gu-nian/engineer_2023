import serial
import time
         
from inference_function.mineral_function import Mineral
from inference_function.station_function import Station
class Interactive_serial(object):
    which_mode = 0
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
            print("Serial Reconnecntion Error")

    # 串口发送移动信息 2停 0左 1右
    def send_mineral_data(self):
        while True:
            time.sleep(0.005)
            if Interactive_serial.which_mode == 0 or Interactive_serial.which_mode == 1:
                try:
                    if Mineral.deviation_x == 0:
                        self.ser.write(('S' + str(2) + str(0) + str(0) + str(0) + 'E').encode("utf-8"))
                    elif Mineral.deviation_x / 100 >= 1:
                        self.ser.write(('S' + str(Mineral.direction) + str(Mineral.deviation_x) + 'E').encode("utf-8"))
                    elif Mineral.deviation_x / 10 >= 1:
                        self.ser.write(('S' + str(Mineral.direction) + str(0) + str(Mineral.deviation_x) + 'E').encode("utf-8"))
                    elif Mineral.deviation_x / 1 >= 1:
                        self.ser.write(('S' + str(Mineral.direction) + str(0) + str(0) + str(Mineral.deviation_x) + 'E').encode("utf-8"))
                    else:
                        self.ser.write(('S' + str(2) + str(0) + str(0) + str(0) + 'E').encode("utf-8"))
                    print("deviation_x: ", Mineral.deviation_x)
                    print("direction: ", Mineral.direction)
                    print(" ")
                except:
                    self.ser.close()
                    print('Serial Send Mineral Data Error')
                    Interactive_serial.serial_connection(self)
    
    # 方向 1， 偏移量 3， pitch角度 2，roll方向 1， roll角度 2
    # 0 负 1 正
    def send_station_data(self):
        while True:
            time.sleep(0.005)
            if Interactive_serial.which_mode == 2:
                try:
                    if Station.deviation_x == 0:
                        if Station.pitch_angle / 10 >= 1:
                            if Station.roll_angle / 10 >= 1:
                                self.ser.write(('S' + str(2) + str(0) + str(0) + str(0) + str(Station.pitch_angle) + str(Station.roll_flag) + str(Station.roll_angle) + 'E').encode("utf-8"))
                            else:
                                self.ser.write(('S' + str(2) + str(0) + str(0) + str(0) + str(Station.pitch_angle) + str(Station.roll_flag) + str(0) + str(Station.roll_angle) + 'E').encode("utf-8"))
                        else:
                            if Station.roll_angle / 10 >= 1:
                                self.ser.write(('S' + str(2) + str(0) + str(0) + str(0) + str(0) + str(Station.pitch_angle) + str(Station.roll_flag) + str(Station.roll_angle) + 'E').encode("utf-8"))
                            else:
                                self.ser.write(('S' + str(2) + str(0) + str(0) + str(0) + str(0) + str(Station.pitch_angle) + str(Station.roll_flag) + str(0) + str(Station.roll_angle) + 'E').encode("utf-8"))
                    
                    elif Station.deviation_x / 100 >= 1:
                        if Station.pitch_angle / 10 >= 1:
                            if Station.roll_angle / 10 >= 1:
                                self.ser.write(('S' + str(Station.direction) + str(Station.deviation_x) + str(Station.pitch_angle) + str(Station.roll_flag) + str(Station.roll_angle) + 'E').encode("utf-8"))
                            else:
                                self.ser.write(('S' + str(Station.direction) + str(Station.deviation_x) + str(Station.pitch_angle) + str(Station.roll_flag) + str(0) + str(Station.roll_angle) + 'E').encode("utf-8"))
                        else:
                            if Station.roll_angle / 10 >= 1:
                                self.ser.write(('S' + str(Station.direction) + str(Station.deviation_x) + str(0) + str(Station.pitch_angle) + str(Station.roll_flag) + str(Station.roll_angle) + 'E').encode("utf-8"))
                            else:
                                self.ser.write(('S' + str(Station.direction) + str(Station.deviation_x) + str(0) + str(Station.pitch_angle) + str(Station.roll_flag) + str(0) + str(Station.roll_angle) + 'E').encode("utf-8"))
                            
                    elif Station.deviation_x / 10 >= 1:
                        if Station.pitch_angle / 10 >= 1:
                            if Station.roll_angle / 10 >= 1:
                                self.ser.write(('S' + str(Station.direction) + str(0) + str(Station.deviation_x) + str(Station.pitch_angle) + str(Station.roll_flag) + str(Station.roll_angle) + 'E').encode("utf-8"))
                            else:
                                self.ser.write(('S' + str(Station.direction) + str(0) + str(Station.deviation_x) + str(Station.pitch_angle) + str(Station.roll_flag) + str(0) + str(Station.roll_angle) + 'E').encode("utf-8"))
                        else:
                            if Station.roll_angle / 10 >= 1:
                                self.ser.write(('S' + str(Station.direction) + str(0) + str(Station.deviation_x) + str(0) + str(Station.pitch_angle) + str(Station.roll_flag) + str(Station.roll_angle) + 'E').encode("utf-8"))
                            else:
                                self.ser.write(('S' + str(Station.direction) + str(0) + str(Station.deviation_x) + str(0) + str(Station.pitch_angle) + str(Station.roll_flag) + str(0) + str(Station.roll_angle) + 'E').encode("utf-8"))

                    elif Station.deviation_x / 1 >= 1:
                        if Station.pitch_angle / 10 >= 1:
                            if Station.roll_angle / 10 >= 1:
                                self.ser.write(('S' + str(Station.direction) + str(0) + str(0) + str(Station.deviation_x) + str(Station.pitch_angle) + str(Station.roll_flag) + str(Station.roll_angle) + 'E').encode("utf-8"))
                            else:
                                self.ser.write(('S' + str(Station.direction) + str(0) + str(0) + str(Station.deviation_x) + str(Station.pitch_angle) + str(Station.roll_flag) + str(0) + str(Station.roll_angle) + 'E').encode("utf-8"))
                        else:
                            if Station.roll_angle / 10 >= 1:
                                self.ser.write(('S' + str(Station.direction) + str(0) + str(0) + str(Station.deviation_x) + str(0) + str(Station.pitch_angle) + str(Station.roll_flag) + str(Station.roll_angle) + 'E').encode("utf-8"))
                            else:
                                self.ser.write(('S' + str(Station.direction) + str(0) + str(0) + str(Station.deviation_x) + str(0) + str(Station.pitch_angle) + str(Station.roll_flag) + str(0) + str(Station.roll_angle) + 'E').encode("utf-8"))
                                
                    else:
                        self.ser.write(('S' + str(2) + str(0) + str(0) + str(0) + str(Station.pitch_angle) + str(Station.roll_flag) + str(Station.roll_angle) + 'E').encode("utf-8"))
                    
                    print("deviation_x: ", Station.deviation_x)
                    print("direction: ", Station.direction)
                    print("pitch_angle: ", Station.pitch_angle)
                    print("roll_flag: ", Station.roll_flag)
                    print("roll_angle: ", Station.roll_angle)
                    print(" ")
                except:
                    self.ser.close()
                    print('Serial Send Station Data Error')
                    Interactive_serial.serial_connection(self)
    
    def send_test_data(self):
        while True:
            a = 0
            for i in range(1000):    
                time.sleep(0.5)
                if a == 5:
                    a = 0
                self.ser.write(('S' + str(a) + str(a) + str(a) + str(a) + str(a) +  str(a) + str(a) + str(a) + str(a) + 'E').encode("utf-8"))
                print(a)
                a += 1
                
    # 串口接收数据
    def receive_data(self):
        while True:
            time.sleep(0.005)
            try:                
                data = self.ser.read(1)
                if data == b'1':
                    Interactive_serial.which_mode = 1
                else:
                    Interactive_serial.which_mode = 2
                print("Interactive_serial.which_mode: ", Interactive_serial.which_mode)
                print(" ")
                self.ser.reset_input_buffer()
            except EOFError:                
                self.ser.close()
                print('Serial Receive Data Error')
                Interactive_serial.serial_connection(self)