from threading import Thread, Lock
import sys
import cv2
import time

def gstreamer_pipeline(
        capture_width=1280,
        capture_height=720,
        display_width=616,
        display_height=820,
        framerate=21,
        flip_method=3,
):
    return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
    )


class Camera:
    def __init__(self, frame_rate: int = 5):
        self.camera = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        self.stopped = False
        self.current_frame = None
        self.frame_mutex = Lock()
        self.frame_rate: float = frame_rate/1000
        self.thread_capture: Thread = Thread()

    def __del__(self):
        self.stop()

    def start(self):
        self.thread_capture = Thread(target=self.update, args=())
        self.thread_capture.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.camera.read()
            if ret:
                with self.frame_mutex:
                    self.current_frame = frame

            time.sleep(self.frame_rate)

    def read(self):
        with self.frame_mutex:
            return self.current_frame

    def stop(self):
        self.stopped = True
        self.thread_capture.join()
        self.camera.release()

    def get_width(self):
        return int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))

    def get_height(self):
        return int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
