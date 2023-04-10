import numpy as np
import cv2


def gstreamer_pipeline(
    capture_width=3280,
    capture_height=2464,
    display_width=820,
    display_height=616,
    framerate=21,
    flip_method=0,
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


cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
while True:
    if cap.isOpened():
        ret, img = cap.read()
        cv2.imshow("Test image", img)
        cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
