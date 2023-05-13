from video_function.video_capture import Video_capture
class Mineral():
    deviation_x = 0
    direction = 0
    target_x = Video_capture.COLS / 2

    def init_serial_data():
        Mineral.deviation_x = 0
        Mineral.direction = 0

    def set_serial_data(deviation_x, direction):
        Mineral.deviation_x = deviation_x
        Mineral.direction = direction
    
    def print_serial_data():
        print("deviation_x: ", Mineral.deviation_x)
        print("direction: ", Mineral.direction)
        print() 